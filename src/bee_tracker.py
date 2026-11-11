"""
Bee Tracker — Detecção e rastreamento de abelha com YOLOv4 (OpenCV DNN).

Desempenho:
- Pós-processamento (decodificação YOLO + NMS) em C++ via cv2.dnn_DetectionModel.
- Leitura/decodificação do vídeo em thread separada (produtor/consumidor).
- Frames pulados descartados com cap.grab() (sem retrieve/cópia).
- Backend GPU (CUDA / CUDA FP16) com detecção automática e fallback para CPU.
- UI atualizada em lotes e resultados em session_state.

Persistência e usuários:
- Contas com senha (PBKDF2) e login obrigatório — ver auth.py.
- Cada análise é salva no SQLite local vinculada ao usuário; o histórico de
  um usuário nunca é visível para outro — ver database.py.
- Eventos registrados em logs/bee_tracker.log e na tabela `logs` — ver
  app_logging.py.
"""

import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta as dtimedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")  # backend sem display — obrigatório em servidor/container
import matplotlib.pyplot as plt

import analytics
import auth
import database
import dataset
import report
import ui
from app_logging import log_event

# --- Configuração (sobrescrevível por variáveis de ambiente) ---
# Por padrão os pesos ficam em <raiz do projeto>/models; o caminho é resolvido
# a partir deste arquivo (src/), então funciona independente do diretório atual.
_BASE_DIR = Path(__file__).resolve().parent.parent
_MODELS_DIR = Path(os.getenv("MODELS_DIR", _BASE_DIR / "models"))
YOLO_CFG = os.getenv("YOLO_CFG", str(_MODELS_DIR / "yolov4-tcc.cfg"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", str(_MODELS_DIR / "yolov4-tcc_best.weights"))
YOLO_NAMES = os.getenv("YOLO_NAMES", str(_MODELS_DIR / "coco.names"))
TARGET_CLASS = os.getenv("TARGET_CLASS", "Abelha")
# Modelos treinados no app (.pt) ficam no volume de dados, p/ persistir.
_TRAINED_DIR = Path(os.path.dirname(database.DB_PATH) or ".") / "models"
# Dataset YOLO externo importado (ex.: imagens originais do YOLOv4) — persistente.
_EXTERNAL_DIR = Path(os.path.dirname(database.DB_PATH) or ".") / "dataset_externo"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

UI_UPDATE_EVERY = 10    # atualiza barra/status a cada N frames processados
QUEUE_SIZE = 32         # profundidade da fila leitor -> inferência

st.set_page_config(page_title="Bee Tracker — YOLOv4", page_icon="🐝",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def _bootstrap():
    """Inicializa banco e logging uma única vez por processo do servidor."""
    database.init_db()
    log_event("app_start", "Aplicação inicializada")
    return True


_bootstrap()


# --- Modelo ---

def cuda_disponivel() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


@st.cache_resource(show_spinner="Carregando modelo YOLO...")
def carregar_classes():
    if not os.path.exists(YOLO_NAMES):
        return None
    with open(YOLO_NAMES, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]


def listar_modelos() -> list[dict]:
    """Modelos de detecção disponíveis: Darknet (.weights) e Ultralytics (.pt).

    Varre a pasta de modelos do projeto e a pasta de modelos treinados (volume
    de dados). Para `X.weights`, usa `X.cfg` se existir, senão o cfg padrão.
    """
    modelos, vistos = [], set()
    for base in (_MODELS_DIR, _TRAINED_DIR):
        try:
            arquivos = sorted(list(base.glob("*.weights")) + list(base.glob("*.pt")))
        except Exception:
            arquivos = []
        for w in arquivos:
            if w.name in vistos:
                continue
            vistos.add(w.name)
            if w.suffix == ".pt":
                modelos.append({"nome": w.name, "weights": str(w), "cfg": "",
                                "tipo": "Ultralytics"})
            else:
                cfg_par = w.with_suffix(".cfg")
                cfg = str(cfg_par) if cfg_par.exists() else YOLO_CFG
                modelos.append({"nome": w.name, "weights": str(w), "cfg": cfg,
                                "tipo": "Darknet"})
    return modelos


class _ModeloUltralytics:
    """Adapta um modelo Ultralytics (.pt) à API .detect() do cv2_DetectionModel.

    Devolve (classes, scores, boxes) com boxes em [x, y, w, h] (px, canto sup-esq),
    no mesmo formato do dnn_DetectionModel — assim o pipeline (processar_video,
    recortes, captura) funciona sem mudanças.
    """

    def __init__(self, yolo, device, imgsz):
        self.yolo = yolo
        self.device = device
        self.imgsz = imgsz

    def detect(self, frame, conf, nms):
        r = self.yolo.predict(frame, conf=conf, iou=nms, imgsz=self.imgsz,
                              device=self.device, verbose=False)[0]
        b = getattr(r, "boxes", None)
        if b is None or len(b) == 0:
            return [], [], []
        cls = b.cls.cpu().numpy().astype(int)
        scr = b.conf.cpu().numpy().astype(float)
        xyxy = b.xyxy.cpu().numpy()
        xywh = np.column_stack([xyxy[:, 0], xyxy[:, 1],
                                xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
        return cls, scr, xywh


def modelo_ativo() -> tuple[str, str]:
    """(cfg, weights) do modelo ativo (definido no admin) ou o padrão."""
    weights = database.get_setting("active_weights") or YOLO_WEIGHTS
    cfg = database.get_setting("active_cfg") or YOLO_CFG
    if not os.path.exists(weights):       # modelo ativo sumiu -> volta ao padrão
        weights, cfg = YOLO_WEIGHTS, YOLO_CFG
    if not os.path.exists(cfg):
        cfg = YOLO_CFG
    return cfg, weights


@st.cache_resource(show_spinner="Carregando modelo YOLO...")
def carregar_modelo(input_size: int, usar_gpu: bool, fp16: bool,
                    cfg_path: str = None, weights_path: str = None):
    """Carrega a rede Darknet (cfg/weights selecionáveis no admin).

    Retorna (DetectionModel, descrição do backend, erro_cuda) — erro_cuda é
    None quando tudo deu certo, ou a mensagem da falha que causou o fallback
    GPU -> CPU (ex.: imagem compilada para outra arquitetura de GPU).
    `cfg_path`/`weights_path` entram na chave de cache, então trocar de modelo
    no admin recarrega automaticamente. Pesos `.pt` carregam via Ultralytics
    (caminho separado do Darknet/OpenCV); se o Ultralytics falhar, cai para o
    Darknet padrão e reporta o motivo.
    """
    wp = weights_path or YOLO_WEIGHTS
    cfg = cfg_path or YOLO_CFG
    erro_cuda = None

    if str(wp).endswith(".pt"):
        try:
            from ultralytics import YOLO
            import torch
            device = 0 if (usar_gpu and torch.cuda.is_available()) else "cpu"
            backend = f"Ultralytics YOLO ({'GPU' if device == 0 else 'CPU'})"
            log_event("model_loaded", f"Ultralytics {os.path.basename(wp)} — {backend}",
                      input_size=input_size)
            return _ModeloUltralytics(YOLO(wp), device, input_size), backend, None
        except Exception as e:
            erro_cuda = ("Ultralytics indisponível: "
                         + (str(e).strip().splitlines()[-1] if str(e) else repr(e)))
            log_event("model_load_failed", erro_cuda, level=logging.ERROR)
            wp, cfg = YOLO_WEIGHTS, YOLO_CFG  # fallback p/ o Darknet padrão

    net = cv2.dnn.readNetFromDarknet(cfg, wp)

    if usar_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CUDA_FP16 if fp16 else cv2.dnn.DNN_TARGET_CUDA
        )
        backend = "GPU (CUDA FP16)" if fp16 else "GPU (CUDA)"
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = "CPU"

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255.0, size=(input_size, input_size),
                         swapRB=True, crop=False)

    # Warm-up: erros de backend CUDA só aparecem no primeiro forward.
    dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    try:
        model.detect(dummy, 0.5, 0.4)
    except Exception as e:
        # ex.: "(-216:No CUDA support) ... check CUDA_ARCH_PTX or CUDA_ARCH_BIN"
        # quando o OpenCV foi compilado para outra compute capability
        erro_cuda = str(e).strip().splitlines()[-1] if str(e) else repr(e)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = "CPU (fallback — backend CUDA falhou)"
        log_event("cuda_fallback", f"Backend CUDA falhou no warm-up: {erro_cuda}",
                  level=logging.WARNING)
        model.detect(dummy, 0.5, 0.4)

    log_event("model_loaded", f"Modelo YOLO carregado — backend: {backend}",
              input_size=input_size)
    return model, backend, erro_cuda


# --- Leitura de vídeo em thread separada ---

class LeitorDeFrames(threading.Thread):
    """Decodifica o vídeo numa thread própria e alimenta uma fila limitada.

    A inferência consome da fila, então decodificação e DNN rodam em paralelo.
    Frames fora do passo de amostragem são descartados com grab() — sem o
    custo de retrieve (conversão de cor + cópia de memória).
    """

    def __init__(self, video_path: str, frame_skip: int,
                 inicio_s: float = 0.0, fim_s: float | None = None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(video_path)
        self.frame_skip = max(1, frame_skip)
        self.fila: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        n_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Recorte temporal: converte segundos -> índices de frame.
        self.inicio_frame = max(0, int(round(inicio_s * self.fps))) if self.fps else 0
        if fim_s is not None and self.fps:
            self.fim_frame = min(n_total - 1, int(round(fim_s * self.fps)))
        else:
            self.fim_frame = n_total - 1
        if self.fim_frame < self.inicio_frame:
            self.fim_frame = self.inicio_frame
        self.total_frames = max(1, self.fim_frame - self.inicio_frame + 1)
        self.aberto = self.cap.isOpened()

    def run(self):
        if self.inicio_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio_frame)
        indice = self.inicio_frame
        try:
            while indice <= self.fim_frame:
                if (indice - self.inicio_frame) % self.frame_skip != 0:
                    if not self.cap.grab():
                        break
                    indice += 1
                    continue
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.fila.put((indice, frame))
                indice += 1
        finally:
            self.fila.put(None)  # sentinela de fim
            self.cap.release()


# --- Processamento ---

@dataclass
class Resultado:
    centroides: np.ndarray      # (M, 2) posições (x, y) — pode ter várias por frame
    indices_frames: np.ndarray  # (M,) frame de origem de cada centróide
    primeiro_frame: np.ndarray
    total_frames: int
    frames_processados: int
    fps_video: float
    tempo_s: float
    fps_processamento: float
    track_ids: np.ndarray = None  # (M,) qual placa/abelha (0..num_alvos-1)
    num_alvos: int = 1
    video_deteccoes: str = None   # caminho do MP4 de conferência (opcional)
    candidatos: list = None       # frames p/ reanotação (opcional)


class RastreadorMultiplo:
    """Associa as detecções de cada frame a trajetórias persistentes (placas).

    Atribuição gulosa pelo vizinho mais próximo da última posição de cada track.
    Adequado quando os alvos (placas) estão espacialmente separados — caso de
    placas de Petri distintas no mesmo vídeo.
    """

    def __init__(self, num_alvos):
        self.num_alvos = max(1, int(num_alvos))
        self.ultima_pos = []  # última (x, y) conhecida de cada track

    def atualizar(self, pontos):
        """Recebe as detecções (já top-N) do frame; devolve [(track_id, (x, y))]."""
        if not pontos:
            return []
        if not self.ultima_pos:
            # IDs estáveis: ordena por posição (esq.->dir., cima->baixo)
            ordenados = sorted(pontos, key=lambda p: (p[0], p[1]))[:self.num_alvos]
            self.ultima_pos = list(ordenados)
            return list(enumerate(ordenados))

        pares = sorted(
            (((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2), di, ti)
            for di, p in enumerate(pontos)
            for ti, q in enumerate(self.ultima_pos)
        )
        pts_usados, tracks_usados, atrib = set(), set(), {}
        for _, di, ti in pares:
            if di in pts_usados or ti in tracks_usados:
                continue
            pts_usados.add(di)
            tracks_usados.add(ti)
            self.ultima_pos[ti] = pontos[di]
            atrib[di] = ti
        # detecções não atribuídas viram novos tracks até atingir o limite
        for di, p in enumerate(pontos):
            if di not in atrib and len(self.ultima_pos) < self.num_alvos:
                self.ultima_pos.append(p)
                atrib[di] = len(self.ultima_pos) - 1
        return [(atrib[di], pontos[di]) for di in sorted(atrib)]


def processar_video(video_path, model, class_names, frame_skip, conf, nms,
                    num_alvos=1, zones=None, ampliar_areas=False, gerar_video=False,
                    inicio_s=0.0, fim_s=None, coletar_candidatos=False,
                    conf_captura=0.15, limiar_revisao=0.5, max_candidatos=200):
    leitor = LeitorDeFrames(video_path, frame_skip, inicio_s, fim_s)
    if not leitor.aberto:
        st.error("Erro ao abrir o arquivo de vídeo.")
        return None

    id_alvo = -1
    if class_names and TARGET_CLASS in class_names:
        id_alvo = class_names.index(TARGET_CLASS)
    elif class_names:
        st.warning(
            f"Classe '{TARGET_CLASS}' não encontrada em '{YOLO_NAMES}'. "
            "Usando a detecção de maior confiança de qualquer classe."
        )

    # Quando há áreas definidas, cada área é uma placa/abelha: a detecção é
    # atribuída à área que a contém e detecções fora de todas as áreas são
    # descartadas como falso positivo. Sem áreas, usa o rastreador por
    # vizinho mais próximo com as N detecções de maior confiança.
    zone_paths = None
    zone_polys = None
    zone_names = []
    if zones:
        from matplotlib.path import Path as _MplPath
        zone_polys = []
        for z in zones:
            pts = z.get("points") or []
            if len(pts) >= 3:
                zone_polys.append(np.asarray(pts, float).reshape(-1, 2))
                zone_names.append(z.get("name") or f"Área {len(zone_names) + 1}")
        if zone_polys:
            zone_paths = [_MplPath(p) for p in zone_polys]
            num_alvos = len(zone_paths)
        else:
            zone_polys = None

    total = max(leitor.total_frames, 1)
    barra = st.progress(0)
    status = st.empty()
    leitor.start()

    centroides, indices_frames, track_ids = [], [], []
    rastreador = RastreadorMultiplo(num_alvos)
    primeiro_frame = None
    processados = 0
    gravador = None
    # Coleta de candidatos para reanotação: amostra frames espaçados e roda um
    # passe de baixa confiança; guarda os fracos/sem-detecção (capado).
    candidatos = []
    proc_estimado = max(1, leitor.total_frames // max(1, frame_skip))
    # Amostra ~60 frames espaçados p/ avaliar candidatos (cada frame pode render
    # vários, um por área); o total fica limitado por max_candidatos.
    passo_cap = max(1, proc_estimado // 60)
    inicio = time.perf_counter()

    while True:
        item = leitor.fila.get()
        if item is None:
            break
        indice, frame = item
        if primeiro_frame is None:
            primeiro_frame = frame.copy()

        # Coletado só quando há vídeo de conferência: detecções brutas do YOLO e
        # as aceitas/rastreadas, para desenhar no frame.
        raw_boxes, aceitos = [], []
        # Confiança da detecção ACEITA em cada área neste frame (para decidir
        # candidatos: área sem detecção / detecção fraca vira candidata).
        score_area_frame = {}

        if zone_paths is not None and ampliar_areas:
            # Detecção ampliada: recorta cada área e detecta ali (a abelha ocupa
            # muito mais pixels). Uma abelha por área; track_id = índice da área.
            for zi, (poly, path) in enumerate(zip(zone_polys, zone_paths)):
                outras = [p for j, p in enumerate(zone_paths) if j != zi]
                res = _detectar_em_recorte(frame, model, conf, nms, id_alvo,
                                           poly, path, outras_paths=outras)
                if res is not None:
                    cx, cy, box, score = res
                    centroides.append((cx, cy))
                    track_ids.append(zi)
                    indices_frames.append(indice)
                    score_area_frame[zi] = score
                    if gerar_video:
                        aceitos.append((cx, cy, zone_names[zi]))
                        raw_boxes.append((*box, conf))
        else:
            classes, scores, boxes = model.detect(frame, conf, nms)

            # todas as detecções da classe alvo: (confiança, cx, cy)
            dets = []
            if len(scores) > 0:
                scores = np.asarray(scores).reshape(-1)
                classes = np.asarray(classes).reshape(-1)
                boxes = np.asarray(boxes).reshape(-1, 4)
                if id_alvo >= 0:
                    mascara = classes == id_alvo
                    scores, boxes = scores[mascara], boxes[mascara]
                for s, (x, y, w, h) in zip(scores, boxes):
                    dets.append((float(s), float(x + w / 2.0), float(y + h / 2.0)))
                    if gerar_video:
                        raw_boxes.append((float(x), float(y), float(w), float(h), float(s)))

            if zone_paths is not None:
                # uma abelha por área: melhor detecção dentro de cada área
                melhor = {}  # índice da área -> (confiança, cx, cy)
                for s, cx, cy in dets:
                    for zi, path in enumerate(zone_paths):
                        if path.contains_point((cx, cy)):
                            if zi not in melhor or s > melhor[zi][0]:
                                melhor[zi] = (s, cx, cy)
                            break  # áreas não se sobrepõem: para na primeira
                for zi, (s, cx, cy) in melhor.items():
                    centroides.append((cx, cy))
                    track_ids.append(zi)
                    indices_frames.append(indice)
                    score_area_frame[zi] = s
                    if gerar_video:
                        aceitos.append((cx, cy, zone_names[zi]))
            else:
                dets.sort(key=lambda d: d[0], reverse=True)
                pontos = [(cx, cy) for _, cx, cy in dets[:num_alvos]]
                for tid, (cx, cy) in rastreador.atualizar(pontos):
                    centroides.append((cx, cy))
                    track_ids.append(tid)
                    indices_frames.append(indice)
                    if gerar_video:
                        aceitos.append((cx, cy, f"Abelha {tid + 1}"))

        # Candidatos para reanotação (frame/recorte LIMPO, antes de desenhar):
        # passe de baixa confiança; guarda os fracos/ausentes. Com áreas, gera
        # UM candidato por área (recorte) — imagem menor, abelha maior.
        def _guardar_candidato(img_bgr, melhor_s, sugeridas, area=None):
            ok_jpg, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok_jpg:
                return
            h0, w0 = img_bgr.shape[:2]
            candidatos.append({
                "frame_index": int(indice),
                "reason": "sem_deteccao" if melhor_s == 0 else "baixa_confianca",
                "pred_conf": melhor_s, "pred_boxes": sugeridas,
                "width": int(w0), "height": int(h0),
                "image_jpg": jpg.tobytes(), "area": area,
            })

        if (coletar_candidatos and len(candidatos) < max_candidatos
                and processados % passo_cap == 0):
            if zone_paths is not None:
                # Um candidato por área cuja detecção ACEITA (na análise) ficou
                # ausente ou fraca — POR ÁREA, não pelo frame todo. O recorte só
                # serve para a imagem e as caixas sugeridas (pré-preenchimento).
                for zi, (poly, path) in enumerate(zip(zone_polys, zone_paths)):
                    if len(candidatos) >= max_candidatos:
                        break
                    sc = score_area_frame.get(zi, 0.0)
                    if sc >= limiar_revisao:
                        continue  # área bem detectada nesta amostra → não é candidata
                    outras = [p for j, p in enumerate(zone_paths) if j != zi]
                    rc = _recorte_candidato(frame, model, conf_captura, nms,
                                            id_alvo, poly, path, outras)
                    if rc is None:
                        continue
                    recorte, _conf_lowpass, sugeridas = rc
                    _guardar_candidato(recorte, sc, sugeridas, zone_names[zi])
            else:
                # sem áreas: candidato do frame inteiro
                cc, sc_c, bx_c = model.detect(frame, conf_captura, nms)
                sugeridas, melhor_s = [], 0.0
                if len(sc_c) > 0:
                    sc_c = np.asarray(sc_c).reshape(-1)
                    cc = np.asarray(cc).reshape(-1)
                    bx_c = np.asarray(bx_c).reshape(-1, 4)
                    if id_alvo >= 0:
                        mc = cc == id_alvo
                        sc_c, bx_c = sc_c[mc], bx_c[mc]
                    for s, (x, y, w, h) in zip(sc_c, bx_c):
                        sugeridas.append([float(x), float(y), float(w), float(h), float(s)])
                    melhor_s = float(np.max(sc_c)) if len(sc_c) else 0.0
                if melhor_s < limiar_revisao:
                    _guardar_candidato(frame, melhor_s, sugeridas, None)

        if gerar_video:
            if gravador is None:
                h0, w0 = frame.shape[:2]
                fps_saida = (leitor.fps or 10.0) / max(frame_skip, 1)
                gravador = _GravadorAnotado(w0, h0, fps_saida)
            _desenhar_anotacoes(frame, zone_polys, raw_boxes, aceitos)
            gravador.escrever(frame)

        processados += 1
        if processados % UI_UPDATE_EVERY == 0:
            rel = indice - leitor.inicio_frame + 1  # progresso dentro do trecho
            barra.progress(min(rel, total) / total)
            fps_proc = processados / max(time.perf_counter() - inicio, 1e-6)
            status.text(
                f"Frame {rel}/{total} — "
                f"{fps_proc:.1f} frames/s ({len(centroides)} detecções)"
            )

    tempo = time.perf_counter() - inicio
    barra.empty()
    status.empty()
    video_conf = gravador.finalizar() if gravador is not None else None

    return Resultado(
        centroides=np.asarray(centroides, dtype=float).reshape(-1, 2),
        indices_frames=np.asarray(indices_frames, dtype=int),
        primeiro_frame=primeiro_frame,
        total_frames=leitor.total_frames,
        frames_processados=processados,
        fps_video=leitor.fps,
        tempo_s=tempo,
        fps_processamento=processados / max(tempo, 1e-6),
        track_ids=np.asarray(track_ids, dtype=int),
        num_alvos=num_alvos,
        video_deteccoes=video_conf,
        candidatos=candidatos if coletar_candidatos else None,
    )


# --- Métricas ---

def distancia_total_px(centroides: np.ndarray) -> float:
    if len(centroides) < 2:
        return 0.0
    difs = np.diff(centroides, axis=0)
    return float(np.hypot(difs[:, 0], difs[:, 1]).sum())


def velocidade_media_px_s(centroides, indices_frames, fps_video) -> float | None:
    if fps_video > 0 and len(indices_frames) > 1:
        duracao = (indices_frames[-1] - indices_frames[0]) / fps_video
        if duracao > 0:
            return distancia_total_px(centroides) / duracao
    return None


# --- Painel de análises (compartilhado por análises novas e salvas) ---

def _mostrar(fig, vazio="Sem dados suficientes para este gráfico."):
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(vazio)


def _recorte_area(fundo, centroides, zona, margem=0.12):
    """Recorta o fundo na caixa da área e desloca os centróides p/ o recorte.

    Usado no detalhamento por placa/área: em vez de mostrar a abelha como um
    ponto minúsculo no frame inteiro, mostra a área ampliada. Distância é
    invariante à translação, então as métricas não mudam.
    """
    if fundo is None or not zona or not zona.get("points"):
        return fundo, centroides
    pts = np.asarray(zona["points"], float).reshape(-1, 2)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    mx, my = (x1 - x0) * margem, (y1 - y0) * margem
    H, W = fundo.shape[:2]
    x0 = int(max(0, x0 - mx)); y0 = int(max(0, y0 - my))
    x1 = int(min(W, x1 + mx)); y1 = int(min(H, y1 + my))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return fundo, centroides
    recorte = fundo[y0:y1, x0:x1].copy()
    desloc = (centroides - np.array([x0, y0])) if len(centroides) else centroides
    return recorte, desloc


def render_zonas(centroides, indices, fundo, fps_video, zones,
                 frame_skip=None, pixels_per_mm=None):
    """Tabela e gráficos de presença/permanência por área de monitoramento."""
    metr_z = analytics.metricas_por_zona(
        centroides, indices, fps_video, zones, frame_skip, pixels_per_mm
    )
    if not metr_z:
        st.info("Nenhuma área definida para esta análise.")
        return

    linhas = []
    for m in metr_z:
        tem_mm = "distancia_dentro_mm" in m
        linha = {
            "Área": m["nome"],
            "Detecções": m["deteccoes_dentro"],
            "% das detecções": f"{m['perc_deteccoes']:.1f}%",
            "Visitas": m["visitas"],
        }
        if "tempo_permanencia" in m:
            linha["Permanência (s)"] = f"{m['tempo_permanencia']:.1f}"
        # Com escala, só mm; sem escala, px.
        if tem_mm:
            linha["Dist. dentro (mm)"] = f"{m['distancia_dentro_mm']:.1f}"
            if "area_mm2" in m:
                linha["Área (mm²)"] = f"{m['area_mm2']:.0f}"
        else:
            linha["Dist. dentro (px)"] = f"{m['distancia_dentro_px']:.0f}"
        linhas.append(linha)
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    st.caption("Permanência = fração do tempo rastreado em que a abelha esteve "
               "dentro da área. Visitas = nº de entradas na área.")

    col1, col2 = st.columns(2)
    with col1:
        _mostrar(analytics.plot_mapa_zonas(centroides, fundo, zones))
    with col2:
        _mostrar(analytics.plot_permanencia_zonas(metr_z))


def render_analytics(centroides, indices, fundo, fps_video, limiar_parada_px,
                     frames_processados=None, frame_skip=None, pixels_per_mm=None,
                     zones=None, area_ref_px2=None):
    """Cartões de métricas + abas de gráficos a partir dos dados brutos.

    `area_ref_px2`: área de referência (px²) para a cobertura da área explorada.
    Quando informada (ex.: área da zona/placa), a cobertura é relativa a ela
    ("% da área"); caso contrário, é relativa ao quadro ("% do quadro").
    """
    metr = analytics.calcular_metricas(
        centroides, indices, fps_video, limiar_parada_px,
        frame_shape=fundo.shape if fundo is not None else None,
        frames_processados=frames_processados, frame_skip=frame_skip,
        pixels_per_mm=pixels_per_mm,
    )
    uv = metr.get("unidade_v", "px/s")
    ut = metr.get("unidade_t", "s")

    # Com escala calibrada, mostramos SÓ as métricas em mm (sem duplicar px).
    tem_mm = "distancia_mm" in metr
    tem_vel_mm = "velocidade_media_mm_s" in metr
    # Cobertura: relativa à área delimitada (se informada) ou ao quadro.
    if area_ref_px2 and area_ref_px2 > 0:
        pct = 100.0 * metr.get("area_explorada_px2", 0.0) / area_ref_px2
        cobertura_txt = f"{pct:.1f}% da área"
    elif metr.get("cobertura_perc") is not None:
        cobertura_txt = f"{metr['cobertura_perc']:.1f}% do quadro"
    else:
        cobertura_txt = None
    with st.expander("📊 Métricas resumidas", expanded=True):
        if tem_mm:
            st.caption(f"Escala calibrada: **{metr['pixels_per_mm']:.2f} px/mm** — "
                       "métricas em milímetros.")
        a, b, c, d = st.columns(4)
        a.metric("Detecções", metr["deteccoes"])
        if tem_mm:
            b.metric("Distância total", f"{metr['distancia_mm']:.1f} mm")
            c.metric("Deslocamento líquido", f"{metr['deslocamento_liquido_mm']:.1f} mm")
        else:
            b.metric("Distância total", f"{metr.get('distancia_px', 0):.0f} px")
            c.metric("Deslocamento líquido",
                     f"{metr.get('deslocamento_liquido_px', 0):.0f} px")
        d.metric("Retilineidade", f"{metr.get('retilineidade', 0):.2f}",
                 help="Deslocamento líquido ÷ distância percorrida. "
                      "1 = linha reta; perto de 0 = trajeto muito sinuoso.")
        e, f, g, h = st.columns(4)
        if tem_mm and tem_vel_mm:
            e.metric("Velocidade média", f"{metr['velocidade_media_mm_s']:.2f} mm/s")
            f.metric("Velocidade máxima", f"{metr['velocidade_max_mm_s']:.2f} mm/s")
        else:
            e.metric("Velocidade média", f"{metr.get('velocidade_media', 0):.1f} {uv}")
            f.metric("Velocidade máxima", f"{metr.get('velocidade_max', 0):.1f} {uv}")
        g.metric("Tempo em movimento", f"{metr.get('perc_movimento', 0):.0f}%",
                 help=f"Fração do tempo com deslocamento ≥ {limiar_parada_px:.0f} px "
                      "entre amostras.")
        if tem_mm:
            h.metric("Área explorada", f"{metr['area_explorada_mm2']:.1f} mm²",
                     delta=cobertura_txt, delta_color="off")
        else:
            h.metric("Área explorada", f"{metr.get('area_explorada_px2', 0):.0f} px²",
                     delta=cobertura_txt, delta_color="off")

        # Detalhamento de tempo (detectado/não detectado, caminhando/parado)
        if "tempo_detectado" in metr:
            st.markdown("**Detalhamento do tempo**")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Tempo detectado", f"{metr['tempo_detectado']:.1f} {ut}",
                      delta=f"{metr.get('perc_detectado', 0):.0f}% dos frames",
                      delta_color="off")
            t2.metric("Tempo não detectado", f"{metr['tempo_nao_detectado']:.1f} {ut}",
                      help="Frames processados em que a abelha não foi detectada.")
            t3.metric("Tempo caminhando", f"{metr.get('tempo_movimento', 0):.1f} {ut}")
            t4.metric("Tempo parado", f"{metr.get('tempo_parado', 0):.1f} {ut}")

    if fundo is None:
        st.info("Sem o frame de fundo não é possível desenhar os gráficos espaciais.")
        return

    with st.expander("📈 Gráficos", expanded=True):
        tem_zonas = bool(zones)
        nomes_abas = ["🗺️ Trajetória", "📈 Velocidade & Tempo", "🌡️ Cobertura espacial"]
        if tem_zonas:
            nomes_abas.append("🎯 Áreas")
        abas = st.tabs(nomes_abas)

        # Gráficos opcionais (configuráveis no menu lateral "📈 Gráficos exibidos")
        ver_posicao = st.session_state.get("graf_posicao", True)
        ver_velocidade = st.session_state.get("graf_velocidade", True)
        ver_histograma = st.session_state.get("graf_histograma", True)

        with abas[0]:
            _mostrar(analytics.plot_trajetoria_tempo(centroides, indices, fundo,
                                                     fps_video, zones=zones),
                     "A trajetória precisa de ao menos 2 detecções.")
            if ver_posicao:
                st.markdown("**Posição X / Y ao longo do tempo**")
                _mostrar(analytics.plot_posicao_tempo(centroides, indices, fps_video))

        with abas[1]:
            if ver_velocidade:
                _mostrar(analytics.plot_velocidade_tempo(centroides, indices, fps_video))
            if ver_histograma:
                col1, col2 = st.columns(2)
                with col1:
                    _mostrar(analytics.plot_distancia_acumulada(centroides, indices, fps_video))
                with col2:
                    _mostrar(analytics.plot_histograma_velocidade(centroides, indices, fps_video))
            else:
                _mostrar(analytics.plot_distancia_acumulada(centroides, indices, fps_video))

        with abas[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Mapa de calor (densidade de permanência)**")
                _mostrar(analytics.plot_heatmap(centroides, fundo, zones=zones))
            with col2:
                st.markdown("**Área explorada (envoltória convexa)**")
                _mostrar(analytics.plot_area_explorada(centroides, fundo),
                         "A área explorada precisa de ao menos 3 detecções.")

        if tem_zonas:
            with abas[3]:
                render_zonas(centroides, indices, fundo, fps_video, zones,
                             frame_skip, pixels_per_mm)


def render_resultados(centroides, indices, track_ids, fundo, fps_video,
                      limiar_parada_px, frames_processados=None, frame_skip=None,
                      pixels_per_mm=None, zones=None, key_prefix=""):
    """Decide entre exibição de placa única ou múltipla a partir dos track_ids."""
    tids = analytics.tracks_unicos(track_ids, centroides)
    if len(tids) <= 1:
        # Com uma área, a cobertura da área explorada é relativa a ela.
        t0 = tids[0]
        area_ref = (analytics._area_poligono_px2(zones[t0]["points"])
                    if zones and t0 < len(zones) else None)
        render_analytics(centroides, indices, fundo, fps_video, limiar_parada_px,
                         frames_processados=frames_processados, frame_skip=frame_skip,
                         pixels_per_mm=pixels_per_mm, zones=zones,
                         area_ref_px2=area_ref)
        return

    # Quando as áreas guiaram o rastreamento, rotula cada placa pelo nome da área.
    def _rotulo(t):
        if zones and t < len(zones):
            return zones[t].get("name") or f"Abelha {t + 1}"
        return f"Abelha {t + 1}"

    rotulos = {t: _rotulo(t) for t in tids}

    with st.expander(f"🐝 Visão geral — {len(tids)} placas/abelhas", expanded=True):
        if fundo is not None:
            _mostrar(analytics.plot_trajetorias_multi(
                centroides, indices, track_ids, fundo, fps_video,
                zones=zones, rotulos=rotulos))

        # Tabela-resumo: uma linha por placa/abelha
        linhas = []
        for t in tids:
            c, idx = analytics.subset_track(centroides, indices, track_ids, t)
            m = analytics.calcular_metricas(
                c, idx, fps_video, limiar_parada_px,
                frame_shape=fundo.shape if fundo is not None else None,
                frames_processados=frames_processados, frame_skip=frame_skip,
                pixels_per_mm=pixels_per_mm,
            )
            tem_mm = "distancia_mm" in m
            linha = {
                "Placa": rotulos[t],
                "Detecções": m.get("deteccoes", 0),
            }
            if tem_mm:
                linha["Distância (mm)"] = f"{m['distancia_mm']:.1f}"
                vel = (f"{m['velocidade_media_mm_s']:.2f} mm/s"
                       if "velocidade_media_mm_s" in m
                       else f"{m.get('velocidade_media', 0):.1f} {m.get('unidade_v', 'px/s')}")
            else:
                linha["Distância (px)"] = f"{m.get('distancia_px', 0):.0f}"
                vel = f"{m.get('velocidade_media', 0):.1f} {m.get('unidade_v', 'px/s')}"
            linha["Vel. média"] = vel
            linha["Tempo mov."] = f"{m.get('perc_movimento', 0):.0f}%"
            linhas.append(linha)
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

    # Detalhamento completo de uma placa escolhida
    opcoes = {rotulos[t]: t for t in tids}
    rotulo = st.selectbox("Ver detalhes de:", list(opcoes.keys()),
                          key=f"{key_prefix}sel_track_detalhe")
    t = opcoes[rotulo]
    c, idx = analytics.subset_track(centroides, indices, track_ids, t)
    st.markdown(f"#### Detalhes — {rotulo}")
    # Recorta o fundo na caixa da área (a abelha ocupa o quadro, não um ponto
    # minúsculo no frame inteiro). Distância é invariante à translação, então as
    # métricas não mudam; zones=None evita redesenhar o contorno em coords do
    # frame inteiro (o recorte já É a área).
    fundo_t, c_t = (_recorte_area(fundo, c, zones[t])
                    if zones and t < len(zones) else (fundo, c))
    # Cobertura relativa à área delimitada (polígono da zona), não ao quadro.
    area_ref = (analytics._area_poligono_px2(zones[t]["points"])
                if zones and t < len(zones) else None)
    render_analytics(c_t, idx, fundo_t, fps_video, limiar_parada_px,
                     frames_processados=frames_processados, frame_skip=frame_skip,
                     pixels_per_mm=pixels_per_mm, zones=None, area_ref_px2=area_ref)


# --- Persistência das análises ---

def salvar_analise(user_id, video_name, video_size, params, res: Resultado) -> int:
    ok, jpg = cv2.imencode(".jpg", res.primeiro_frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    # Distância agregada = soma das distâncias de cada placa/abelha
    tids = analytics.tracks_unicos(res.track_ids)
    distancia_total = sum(
        distancia_total_px(analytics.subset_track(
            res.centroides, res.indices_frames, res.track_ids, t)[0])
        for t in tids
    )
    analysis_id = database.save_analysis(
        user_id=user_id,
        video_name=video_name,
        video_size_bytes=video_size,
        params=params,
        metrics=dict(
            fps_video=res.fps_video,
            total_frames=res.total_frames,
            frames_processed=res.frames_processados,
            detections=len(res.centroides),
            distance_px=distancia_total,
            avg_speed_px_s=velocidade_media_px_s(
                res.centroides, res.indices_frames, res.fps_video
            ),
            processing_time_s=res.tempo_s,
            fps_processing=res.fps_processamento,
            num_targets=res.num_alvos,
        ),
        centroids_blob=database.ndarray_to_blob(res.centroides),
        frame_indices_blob=database.ndarray_to_blob(res.indices_frames),
        first_frame_jpg=jpg.tobytes() if ok else None,
        track_ids_blob=database.ndarray_to_blob(res.track_ids)
        if res.track_ids is not None else None,
    )
    # Não inclui o JSON das zonas no log (verboso); registra só a contagem.
    log_params = {k: v for k, v in params.items() if k != "zones"}
    num_zonas = len(json.loads(params["zones"])) if params.get("zones") else 0
    log_event("analysis_completed",
              f"Análise #{analysis_id} de '{video_name}' concluída",
              user_id=user_id, analysis_id=analysis_id,
              deteccoes=len(res.centroides), tempo_s=round(res.tempo_s, 1),
              num_zonas=num_zonas, **log_params)
    return analysis_id


def csv_da_analise(centroides, indices_frames, track_ids=None) -> str:
    dados = {"frame": indices_frames, "x": centroides[:, 0], "y": centroides[:, 1]}
    if track_ids is not None and len(np.asarray(track_ids).reshape(-1)):
        dados = {"placa": np.asarray(track_ids).reshape(-1) + 1, **dados}
    return pd.DataFrame(dados).to_csv(index=False)


# --- Telas ---

def tela_login():
    ui.inject_css()
    _, centro, _ = st.columns([1, 1.4, 1])
    with centro:
        ui.header("Bee Tracker", "Detecção e rastreamento de abelhas com YOLOv4")
        st.caption("Entre com a sua conta para ver e criar análises. "
                   "Cada usuário enxerga apenas as próprias análises.")

        aba_entrar, aba_registrar = st.tabs(["🔑 Entrar", "✨ Criar conta"])

        with aba_entrar:
            with st.form("form_login"):
                usuario = st.text_input("Usuário")
                senha = st.text_input("Senha", type="password")
                if st.form_submit_button("Entrar", type="primary",
                                         width="stretch"):
                    user, erro = auth.login(usuario, senha)
                    if user:
                        st.session_state["user_id"] = user["id"]
                        st.session_state["username"] = user["username"]
                        st.session_state["role"] = user.get("role", "user")
                        st.rerun()
                    else:
                        st.error(erro)

        with aba_registrar:
            with st.form("form_registro"):
                usuario = st.text_input("Usuário", help="Mínimo de 3 caracteres")
                senha = st.text_input("Senha", type="password",
                                      help="Mínimo de 6 caracteres")
                confirma = st.text_input("Confirmar senha", type="password")
                if st.form_submit_button("Criar conta", width="stretch"):
                    if senha != confirma:
                        st.error("As senhas não conferem.")
                    else:
                        ok, msg = auth.register(usuario, senha)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)


def exibir_resultados(res: Resultado, frame_skip: int, limiar_parada_px: float,
                      pixels_per_mm: float | None = None, zones=None):
    if len(res.centroides) == 0:
        st.warning("Nenhuma abelha detectada nos frames processados com os parâmetros atuais.")
        st.info("Dicas: reduza o limiar de confiança ou o salto de frames na barra lateral.")
        return

    st.caption(
        f"Processado em {res.tempo_s:.1f} s a {res.fps_processamento:.1f} frames/s — "
        f"{res.frames_processados}/{res.total_frames} frames (1 a cada {frame_skip})."
    )
    st.download_button(
        "⬇️ Baixar trajetória (CSV)",
        data=csv_da_analise(res.centroides, res.indices_frames, res.track_ids),
        file_name="trajetoria.csv", mime="text/csv",
    )
    render_resultados(res.centroides, res.indices_frames, res.track_ids,
                      res.primeiro_frame, res.fps_video, limiar_parada_px,
                      frames_processados=res.frames_processados,
                      frame_skip=frame_skip, pixels_per_mm=pixels_per_mm, zones=zones,
                      key_prefix="nova_")


def exibir_analise_salva(analise: dict, user_id: int, limiar_parada_px: float):
    centroides = database.blob_to_ndarray(analise["centroids"])
    indices = database.blob_to_ndarray(analise["frame_indices"])
    track_ids = (database.blob_to_ndarray(analise["track_ids"])
                 if "track_ids" in analise.keys() and analise["track_ids"] else None)
    fundo = None
    if analise["first_frame_jpg"]:
        fundo = cv2.imdecode(
            np.frombuffer(analise["first_frame_jpg"], np.uint8), cv2.IMREAD_COLOR
        )

    escala_salva = analise["pixels_per_mm"] if "pixels_per_mm" in analise.keys() else None
    zonas_salvas = None
    if "zones" in analise.keys() and analise["zones"]:
        try:
            zonas_salvas = json.loads(analise["zones"])
        except (ValueError, TypeError):
            zonas_salvas = None
    st.markdown(f"### Análise #{analise['id']} — {analise['video_name']}")
    st.caption(
        f"Processada em {analise['created_at']} | backend: {analise['backend']} | "
        f"frame skip: {analise['frame_skip']} | rede: {analise['input_size']}px | "
        f"confiança: {analise['conf_threshold']} | NMS: {analise['nms_threshold']} | "
        f"tempo de processamento: {analise['processing_time_s']:.1f} s"
        + (f" | escala: {escala_salva:.2f} px/mm" if escala_salva else "")
        + (f" | {len(zonas_salvas)} área(s)" if zonas_salvas else "")
    )

    if centroides is not None and len(centroides) > 0:
        st.download_button(
            "⬇️ Baixar trajetória (CSV)",
            data=csv_da_analise(centroides, indices, track_ids),
            file_name=f"analise_{analise['id']}_trajetoria.csv",
            mime="text/csv",
        )
        render_resultados(centroides, indices, track_ids, fundo,
                          analise["fps_video"], limiar_parada_px,
                          frames_processados=analise["frames_processed"],
                          frame_skip=analise["frame_skip"],
                          pixels_per_mm=escala_salva, zones=zonas_salvas,
                          key_prefix=f"hist{analise['id']}_")
    else:
        st.info("Esta análise não registrou detecções.")

    st.markdown("---")
    if st.button("🗑️ Excluir esta análise", key=f"excluir_{analise['id']}"):
        database.delete_analysis(analise["id"], user_id)
        log_event("analysis_deleted", f"Análise #{analise['id']} excluída",
                  user_id=user_id, analysis_id=analise["id"])
        st.rerun()


def render_historico(user_id: int, username: str, limiar_parada_px: float):
    analises = database.list_analyses(user_id)
    if not analises:
        st.info("Você ainda não tem análises salvas. "
                "Processe um vídeo na aba 'Nova Análise'.")
    else:
        st.caption(f"{len(analises)} análise(s) — visíveis apenas para a sua conta.")
        df = pd.DataFrame(analises)[[
            "id", "created_at", "video_name", "detections",
            "distance_px", "frame_skip", "backend", "processing_time_s",
        ]].rename(columns={
            "id": "ID", "created_at": "Data", "video_name": "Vídeo",
            "detections": "Detecções", "distance_px": "Distância (px)",
            "frame_skip": "Frame skip", "backend": "Backend",
            "processing_time_s": "Tempo (s)",
        })
        st.dataframe(df, hide_index=True, width="stretch")

        # --- Relatório geral consolidado ---
        st.markdown("#### Relatório geral")
        col_csv, col_xlsx, col_pdf = st.columns(3)
        with col_csv:
            st.download_button(
                "⬇️ Resumo (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resumo_analises.csv", mime="text/csv",
                width="stretch",
            )
        with col_xlsx:
            if st.button("📊 Gerar Excel", width="stretch"):
                try:
                    st.session_state["relatorio_xlsx"] = report.build_excel(
                        database.list_analyses_full(user_id), username,
                        limiar_parada_px)
                    log_event("report_excel_generated",
                              f"Relatório Excel gerado ({len(analises)} análises)",
                              user_id=user_id)
                except Exception as e:
                    st.error(f"Falha ao gerar Excel: {e}")
            if st.session_state.get("relatorio_xlsx"):
                st.download_button(
                    "⬇️ Baixar (Excel)",
                    data=st.session_state["relatorio_xlsx"],
                    file_name="relatorio_bee_tracker.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
        with col_pdf:
            if st.button("📄 Gerar relatório (PDF)", width="stretch"):
                with st.spinner("Montando relatório com métricas e gráficos..."):
                    try:
                        st.session_state["relatorio_pdf"] = report.build_pdf(
                            database.list_analyses_full(user_id), username,
                            limiar_parada_px,
                        )
                        log_event("report_generated",
                                  f"Relatório PDF gerado ({len(analises)} análises)",
                                  user_id=user_id)
                    except Exception as e:
                        st.error(f"Falha ao gerar relatório: {e}")
                        log_event("report_failed", f"Erro no relatório: {e}",
                                  user_id=user_id, level=logging.ERROR)
        if st.session_state.get("relatorio_pdf"):
            st.download_button(
                "⬇️ Baixar relatório (PDF)",
                data=st.session_state["relatorio_pdf"],
                file_name="relatorio_bee_tracker.pdf", mime="application/pdf",
            )

        st.markdown("---")
        opcoes = {
            f"#{a['id']} — {a['video_name']} ({a['created_at']})": a["id"]
            for a in analises
        }
        rotulo = st.selectbox("Abrir análise", list(opcoes.keys()))
        analise = database.get_analysis(opcoes[rotulo], user_id)
        if analise:
            exibir_analise_salva(analise, user_id, limiar_parada_px)

    with st.expander("📜 Histórico de atividade da conta"):
        eventos = database.list_logs(user_id=user_id, limit=100)
        if eventos:
            st.dataframe(pd.DataFrame(eventos), hide_index=True,
                         width="stretch")
        else:
            st.caption("Sem eventos registrados.")


def _patch_canvas_compat():
    """Restaura streamlit.elements.image.image_to_url, removido no Streamlit ≥1.49.

    O streamlit-drawable-canvas 0.9.3 chama essa função para obter a URL da
    imagem de fundo do canvas. Reimplementamos registrando a imagem no
    MediaFileManager do Streamlit (o mesmo mecanismo do st.image) e devolvendo
    a URL '/media/...', que o frontend do canvas sabe carregar. Um data-URL
    NÃO funciona aqui — o componente prefixa a URL com a origem do servidor,
    o que quebraria um 'data:'. Por isso a imagem não aparecia.
    """
    try:
        import streamlit.elements.image as st_image_mod
    except Exception:
        return
    if hasattr(st_image_mod, "image_to_url"):
        return

    import io as _io
    from PIL import Image as _PILImage
    import streamlit.elements.lib.image_utils as _iu

    def image_to_url(image, width=None, clamp=False, channels="RGB",
                     output_format="PNG", image_id=""):
        pil = (image if isinstance(image, _PILImage.Image)
               else _PILImage.fromarray(np.asarray(image).astype("uint8")))
        buf = _io.BytesIO()
        pil.save(buf, format="PNG")
        if _iu.runtime.exists():
            return _iu.runtime.get_instance().media_file_mgr.add(
                buf.getvalue(), "image/png", image_id or "drawable-canvas-bg"
            )
        return ""

    st_image_mod.image_to_url = image_to_url


def _primeiro_frame(video_path):
    """Primeiro frame do vídeo, cacheado por sessão.

    Evita reabrir o vídeo a cada rerun do Streamlit — a releitura repetida de
    arquivos grandes às vezes falhava de forma intermitente, deixando o canvas
    sem imagem de fundo.
    """
    cache = st.session_state.get("_frame_cache")
    if cache and cache[0] == video_path:
        return cache[1]
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    st.session_state["_frame_cache"] = (video_path, frame)
    return frame


def _duracao_video(video_path):
    """Duração do vídeo em segundos (cacheada por sessão)."""
    cache = st.session_state.get("_dur_cache")
    if cache and cache[0] == video_path:
        return cache[1]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    dur = (n / fps) if fps > 0 else 0.0
    st.session_state["_dur_cache"] = (video_path, dur)
    return dur


def _seg_para_time(s):
    """Segundos (número) -> datetime.time (hh:mm:ss)."""
    s = int(s)
    return dtime(s // 3600, (s % 3600) // 60, s % 60)


def _time_para_seg(t):
    """datetime.time -> segundos (int)."""
    return t.hour * 3600 + t.minute * 60 + t.second


def _fmt_hms(s):
    """Segundos -> 'hh:mm:ss'."""
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def ferramenta_calibracao(video_path):
    """Permite desenhar uma linha sobre objeto de medida conhecida -> px/mm."""
    try:
        _patch_canvas_compat()
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception:
        st.info("Componente de desenho indisponível. Informe a escala manualmente:")
        st.session_state["pixels_per_mm"] = st.number_input(
            "Escala (pixels por mm)", min_value=0.0,
            value=float(st.session_state.get("pixels_per_mm", 0.0)),
            step=0.1, format="%.2f",
        )
        return

    frame = _primeiro_frame(video_path)
    if frame is None:
        st.warning("Não foi possível ler o primeiro frame para calibração.")
        return

    altura, largura = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    st.markdown(
        "Desenhe **uma linha reta** (clique e arraste) sobre um objeto cujo "
        "comprimento real você conhece — ex.: o diâmetro da placa de Petri — "
        "e informe esse comprimento em milímetros."
    )

    canvas_w = max(1, min(largura, 700))
    canvas_h = int(altura * canvas_w / largura)
    col_canvas, col_input = st.columns([3, 1])
    with col_canvas:
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=3, stroke_color="#FF0000",
            background_image=pil, update_streamlit=True,
            height=canvas_h, width=canvas_w, drawing_mode="line", key="canvas_calib",
        )
    with col_input:
        comprimento_mm = st.number_input("Comprimento real (mm)", min_value=0.01,
                                         value=100.0, step=0.1, key="comprimento_mm")
        if st.button("Calcular escala"):
            objetos = (canvas.json_data or {}).get("objects") if canvas else None
            linhas = [o for o in (objetos or []) if o.get("type") == "line"]
            if not linhas:
                st.error("Desenhe uma linha reta sobre o objeto primeiro.")
            else:
                o = linhas[-1]
                # O comprimento (x2-x1, y2-y1) independe de left/top: o
                # deslocamento entre extremos é invariante ao centro do objeto.
                comp_canvas = float(np.hypot(o["x2"] - o["x1"], o["y2"] - o["y1"]))
                comp_imagem = comp_canvas * (largura / canvas_w)  # canvas -> resolução real
                if comp_imagem > 0:
                    escala = comp_imagem / comprimento_mm
                    st.session_state["pixels_per_mm"] = escala
                    st.success(f"Escala: {escala:.2f} px/mm "
                               f"({comp_imagem:.0f} px = {comprimento_mm:.2f} mm)")
                else:
                    st.error("Linha de comprimento zero. Desenhe novamente.")

    escala = st.session_state.get("pixels_per_mm", 0.0)
    if escala > 0:
        st.caption(f"Escala atual: **{escala:.2f} px/mm**. "
                   "As métricas em mm aparecerão nos resultados.")
        if st.button("Limpar escala"):
            st.session_state["pixels_per_mm"] = 0.0
            st.rerun()
    else:
        st.caption("Sem escala definida — resultados apenas em pixels.")


def _poligono_do_objeto(obj, escala, n_circulo=40):
    """Extrai os vértices (em px da imagem) de um objeto do canvas (fabric.js).

    Suporta polígonos, círculos/elipses e objetos 'path'. Um círculo vira um
    polígono de aproximação (`n_circulo` lados) — assim toda a lógica de zona
    (ponto-em-polígono, área, desenho) funciona sem casos especiais.
    Coordenadas do canvas são multiplicadas por `escala` (canvas -> imagem).
    """
    pts = []
    tipo = obj.get("type")
    if tipo in ("circle", "Circle"):
        left, top = obj.get("left", 0.0), obj.get("top", 0.0)
        raio = obj.get("radius", 0.0)
        sx, sy = obj.get("scaleX", 1.0), obj.get("scaleY", 1.0)
        rx, ry = raio * sx, raio * sy
        # O CircleTool do streamlit-drawable-canvas usa o ponto do clique como
        # ORIGEM (originX="left", originY="center" → ponto na circunferência) e
        # ROTACIONA o círculo por `angle` (graus) na direção do arraste. Logo o
        # centro NÃO é (left+rx, top): é a origem mais o vetor (offset_local)
        # girado por `angle`. Honramos origin + rotação (fórmula geral do fabric).
        ox = obj.get("originX", "left")
        oy = obj.get("originY", "center")
        off_x = 0.0 if ox == "center" else (-rx if ox == "right" else rx)
        off_y = 0.0 if oy == "center" else (-ry if oy == "bottom" else ry)
        ang = np.radians(obj.get("angle", 0.0) or 0.0)
        ca, sa = np.cos(ang), np.sin(ang)
        cx = left + off_x * ca - off_y * sa
        cy = top + off_x * sa + off_y * ca
        for k in range(n_circulo):
            th = 2.0 * np.pi * k / n_circulo
            ex, ey = rx * np.cos(th), ry * np.sin(th)  # ponto na elipse (eixos locais)
            pts.append([(cx + ex * ca - ey * sa) * escala,
                        (cy + ex * sa + ey * ca) * escala])
    elif tipo in ("polygon", "Polygon") and obj.get("points"):
        left, top = obj.get("left", 0.0), obj.get("top", 0.0)
        sx, sy = obj.get("scaleX", 1.0), obj.get("scaleY", 1.0)
        xs = [p["x"] for p in obj["points"]]
        ys = [p["y"] for p in obj["points"]]
        minx, miny = min(xs), min(ys)
        for p in obj["points"]:
            pts.append([(left + (p["x"] - minx) * sx) * escala,
                        (top + (p["y"] - miny) * sy) * escala])
    elif obj.get("path"):
        for seg in obj["path"]:
            if len(seg) >= 3 and isinstance(seg[1], (int, float)):
                pts.append([seg[1] * escala, seg[2] * escala])
    return pts


def _circulo_para_poligono(cx, cy, r, n=40):
    """Converte um círculo (px da imagem) em polígono de `n` lados."""
    return [[float(cx + r * np.cos(2 * np.pi * k / n)),
             float(cy + r * np.sin(2 * np.pi * k / n))] for k in range(n)]


def _circulos_para_canvas(circulos, escala):
    """Monta um initial_drawing (fabric.js) com os círculos para o canvas.

    `circulos` = lista de (cx, cy, r) em px da IMAGEM; dividimos por `escala`
    para o espaço do canvas. originX/originY="center" faz `left`/`top` ser o
    centro — o parser `_poligono_do_objeto` lê de volta sem ambiguidade — e o
    usuário pode mover/excluir/redimensionar cada forma antes de salvar.
    """
    objetos = [
        {"type": "circle",
         "left": cx / escala, "top": cy / escala, "radius": r / escala,
         "originX": "center", "originY": "center", "angle": 0,
         "scaleX": 1, "scaleY": 1, "strokeWidth": 2,
         "stroke": "#FFA500", "fill": "rgba(255,165,0,0.25)"}
        for cx, cy, r in circulos
    ]
    return {"version": "4.4.0", "objects": objetos}


def _detectar_placas_hough(frame, n_placas, param2=30,
                           raio_min_frac=0.05, raio_max_frac=0.25):
    """Detecta até `n_placas` círculos (placas de Petri) via transformada de Hough.

    Roda em escala de cinza suavizada; devolve lista de (cx, cy, r) em px da
    imagem, ordenada pela força do acumulador (círculos mais votados primeiro).
    `param2` menor detecta mais círculos (e mais falsos). Raios em fração do
    menor lado do frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    lado = min(frame.shape[:2])
    r_min = max(5, int(raio_min_frac * lado))
    r_max = max(r_min + 1, int(raio_max_frac * lado))
    min_dist = max(10, int(r_min * 1.5))
    circ = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
        param1=120, param2=max(1, int(param2)),
        minRadius=r_min, maxRadius=r_max,
    )
    if circ is None:
        return []
    circ = np.round(circ[0]).astype(int)  # HOUGH_GRADIENT já ordena por votos
    return [(int(x), int(y), int(r)) for x, y, r in circ[:n_placas]]


def _detectar_em_recorte(frame, model, conf, nms, id_alvo, poligono, path,
                         margem=0.15, outras_paths=None):
    """Detecta a abelha dentro do recorte (ampliado) de uma área.

    Recorta a caixa que envolve o polígono da área COM uma folga lateral
    (`margem`) e roda a YOLO só nesse recorte — que o dnn_DetectionModel
    reamostra para o tamanho da rede, fazendo a abelha ocupar muito mais pixels
    (ganho em vídeo de baixa resolução). Como o recorte inclui as laterais,
    pode aparecer a abelha de uma PLACA ADJACENTE: por isso só aceitamos a
    detecção cujo centro está DENTRO desta área e FORA de qualquer área vizinha
    (`outras_paths`) — caso contrário é descartada como falso positivo.
    Devolve (cx, cy, box, score) da melhor detecção válida (box em coords do
    frame) ou None.
    """
    H, W = frame.shape[:2]
    arr = np.asarray(poligono, float).reshape(-1, 2)
    x0f, y0f, x1f, y1f = arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()
    mx, my = (x1f - x0f) * margem, (y1f - y0f) * margem
    x0 = int(max(0, np.floor(x0f - mx)))
    y0 = int(max(0, np.floor(y0f - my)))
    x1 = int(min(W, np.ceil(x1f + mx)))
    y1 = int(min(H, np.ceil(y1f + my)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    classes, scores, boxes = model.detect(frame[y0:y1, x0:x1], conf, nms)
    if len(scores) == 0:
        return None
    scores = np.asarray(scores).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    boxes = np.asarray(boxes).reshape(-1, 4)
    if id_alvo >= 0:
        m = classes == id_alvo
        scores, boxes = scores[m], boxes[m]
    melhor = None
    for s, (x, y, w, h) in zip(scores, boxes):
        cx, cy = x0 + x + w / 2.0, y0 + y + h / 2.0
        if not path.contains_point((cx, cy)):
            continue  # fora desta área (na folga lateral) → ignora
        if outras_paths and any(p.contains_point((cx, cy)) for p in outras_paths):
            continue  # caiu numa placa adjacente → falso positivo
        if melhor is None or s > melhor[0]:
            melhor = (float(s), cx, cy,
                      (float(x0 + x), float(y0 + y), float(w), float(h)))
    return (melhor[1], melhor[2], melhor[3], melhor[0]) if melhor else None


def _recorte_candidato(frame, model, conf_captura, nms, id_alvo, poligono, path,
                       outras_paths=None, margem=0.15):
    """Recorta a área e roda um passe de baixa confiança SÓ no recorte.

    Para coletar candidatos a reanotação por ÁREA (imagem menor, abelha maior).
    Devolve (recorte_bgr, melhor_conf, caixas_sugeridas) — caixas em coords do
    RECORTE [[x,y,w,h,score], ...], apenas das detecções com centro dentro desta
    área e fora das vizinhas. None se o recorte for inválido.
    """
    H, W = frame.shape[:2]
    arr = np.asarray(poligono, float).reshape(-1, 2)
    x0f, y0f, x1f, y1f = arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()
    mx, my = (x1f - x0f) * margem, (y1f - y0f) * margem
    x0 = int(max(0, np.floor(x0f - mx)))
    y0 = int(max(0, np.floor(y0f - my)))
    x1 = int(min(W, np.ceil(x1f + mx)))
    y1 = int(min(H, np.ceil(y1f + my)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    recorte = frame[y0:y1, x0:x1]
    classes, scores, boxes = model.detect(recorte, conf_captura, nms)
    sugeridas, melhor = [], 0.0
    if len(scores) > 0:
        scores = np.asarray(scores).reshape(-1)
        classes = np.asarray(classes).reshape(-1)
        boxes = np.asarray(boxes).reshape(-1, 4)
        if id_alvo >= 0:
            m = classes == id_alvo
            scores, boxes = scores[m], boxes[m]
        for s, (x, y, w, h) in zip(scores, boxes):
            cxf, cyf = x0 + x + w / 2.0, y0 + y + h / 2.0  # centro no frame
            if not path.contains_point((cxf, cyf)):
                continue
            if outras_paths and any(p.contains_point((cxf, cyf)) for p in outras_paths):
                continue
            sugeridas.append([float(x), float(y), float(w), float(h), float(s)])
            melhor = max(melhor, float(s))
    return recorte.copy(), melhor, sugeridas


class _GravadorAnotado:
    """Grava frames BGR num MP4 H.264 (browser-friendly) via ffmpeg por pipe.

    Encode com libx264/yuv420p (reproduz em qualquer navegador); `pad` garante
    dimensões pares (exigência do yuv420p). Falhas de escrita são silenciadas
    para nunca interromper o processamento.
    """

    def __init__(self, largura, altura, fps):
        self.path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        self.ok = False
        try:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{largura}x{altura}", "-r", f"{max(fps, 1.0):.4f}", "-i", "-",
                 "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE)
            self.ok = True
        except Exception:
            self.proc = None

    def escrever(self, frame_bgr):
        if not self.ok:
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except (BrokenPipeError, OSError):
            self.ok = False

    def finalizar(self):
        if self.proc is None:
            return None
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()
        return self.path if os.path.exists(self.path) else None


def _desenhar_anotacoes(img, zone_polys, raw_boxes, aceitos):
    """Desenha áreas (laranja), detecções brutas do YOLO (amarelo, com confiança)
    e detecções aceitas/rastreadas (verde, com rótulo) sobre `img` (BGR)."""
    if zone_polys:
        for poly in zone_polys:
            cv2.polylines(img, [poly.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 165, 255), 2)
    for (x, y, w, h, s) in raw_boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 255, 255), 1)
        cv2.putText(img, f"{s:.2f}", (int(x), int(y) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    for (cx, cy, label) in aceitos:
        cv2.circle(img, (int(cx), int(cy)), 6, (0, 255, 0), -1)
        if label:
            cv2.putText(img, label, (int(cx) + 7, int(cy) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def ferramenta_zonas(video_path):
    """Permite desenhar polígonos sobre o frame para definir áreas de monitoramento."""
    try:
        _patch_canvas_compat()
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception:
        st.info("Componente de desenho indisponível — não é possível definir áreas "
                "nesta instalação.")
        return

    frame = _primeiro_frame(video_path)
    if frame is None:
        st.warning("Não foi possível ler o primeiro frame para definir áreas.")
        return

    altura, largura = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Após "Detectar placas", já entra no modo de ajuste (definido antes do widget,
    # pois não se pode alterar o session_state de um widget já instanciado).
    if st.session_state.pop("zona_forcar_ajuste", False):
        st.session_state["zona_formato"] = "✏️ Ajustar (mover/excluir)"

    formato = st.radio(
        "Formato da área", ["Polígono", "Círculo / elipse", "✏️ Ajustar (mover/excluir)"],
        horizontal=True, key="zona_formato",
    )
    modo = {"Polígono": "polygon", "Círculo / elipse": "circle"}.get(formato, "transform")
    if modo == "polygon":
        st.markdown(
            "Desenhe **um polígono por área**: clique em cada vértice e **clique de "
            "novo no primeiro ponto** para fechar. Depois clique em *Salvar áreas*."
        )
    elif modo == "circle":
        st.markdown(
            "Desenhe **um círculo por área**: clique e arraste para definir o "
            "tamanho. Depois clique em *Salvar áreas*."
        )
    else:
        st.markdown(
            "Modo **ajuste**: clique numa forma para selecioná-la, arraste para "
            "mover ou redimensionar e use o ícone de **lixeira** (barra do canvas) "
            "para excluir. Útil para corrigir as placas detectadas automaticamente."
        )

    canvas_w = max(1, min(largura, 700))
    canvas_h = int(altura * canvas_w / largura)
    escala = largura / canvas_w

    # Placas detectadas por Hough são injetadas no canvas para ajuste manual.
    circ_hough = st.session_state.get("zonas_hough_circ")
    init_draw = _circulos_para_canvas(circ_hough, escala) if circ_hough else None

    col_canvas, col_input = st.columns([3, 1])
    with col_canvas:
        canvas = st_canvas(
            fill_color="rgba(255,165,0,0.25)", stroke_width=2, stroke_color="#FFA500",
            background_image=pil, update_streamlit=True, initial_drawing=init_draw,
            height=canvas_h, width=canvas_w, drawing_mode=modo, key="canvas_zonas",
        )
    with col_input:
        prefixo = st.text_input("Nome base das áreas", value="Área", key="zonas_prefixo")
        if st.button("Salvar áreas desenhadas"):
            objetos = (canvas.json_data or {}).get("objects") if canvas else None
            poligonos = [p for o in (objetos or [])
                         if len(p := _poligono_do_objeto(o, escala)) >= 3]
            if not poligonos:
                st.error("Desenhe ao menos uma área (polígono ou círculo) antes de salvar.")
            else:
                st.session_state["zones"] = [
                    {"name": f"{prefixo} {i + 1}", "points": p}
                    for i, p in enumerate(poligonos)
                ]
                st.success(f"{len(poligonos)} área(s) salvas.")
                st.rerun()

        st.markdown("---")
        st.markdown("**Detecção automática (Hough)**")
        n_placas_auto = st.number_input(
            "Quantas placas detectar", min_value=1, max_value=30, value=10,
            key="hough_n",
            help="Detecta automaticamente os N círculos mais nítidos (placas de "
                 "Petri) por transformada de Hough e cria uma área para cada um.",
        )
        with st.expander("Parâmetros do Hough"):
            param2 = st.slider(
                "Sensibilidade (param2)", 10, 100, 30, key="hough_p2",
                help="Menor = detecta mais círculos (inclui mais falsos); "
                     "maior = só os mais nítidos.",
            )
            raio_min = st.slider("Raio mínimo (% do lado)", 1, 30, 5,
                                 key="hough_rmin") / 100.0
            raio_max = st.slider("Raio máximo (% do lado)", 5, 60, 25,
                                 key="hough_rmax") / 100.0
        if st.button("🔍 Detectar placas"):
            circulos = _detectar_placas_hough(
                frame, int(n_placas_auto), param2, raio_min, raio_max)
            if not circulos:
                st.warning("Nenhum círculo detectado. Ajuste os parâmetros do Hough "
                           "(reduza a sensibilidade ou ajuste a faixa de raios).")
            else:
                # Injeta no canvas para ajuste; o usuário corrige e clica em Salvar.
                st.session_state["zonas_hough_circ"] = circulos
                st.session_state["zona_forcar_ajuste"] = True  # já entra no modo ajustar
                st.success(f"{len(circulos)} placa(s) detectada(s) — ajuste no canvas "
                           "(modo *Ajustar* move/exclui) e clique em *Salvar áreas*.")
                st.rerun()

    zonas = st.session_state.get("zones") or []
    if zonas:
        # O canvas acima já mostra as áreas sobre o frame — não repetir o preview.
        st.caption(f"**{len(zonas)} área(s) definidas.** Renomeie abaixo se quiser:")
        st.caption("Nomes das áreas:")
        cols_nomes = st.columns(5)
        for i, z in enumerate(zonas):
            z["name"] = cols_nomes[i % 5].text_input(
                f"Área {i + 1}", value=z["name"], key=f"zona_nome_{i}")
        if st.button("Limpar áreas"):
            st.session_state.pop("zones", None)
            st.session_state.pop("zonas_hough_circ", None)
            st.rerun()
    else:
        st.caption("Nenhuma área definida — o relatório terá apenas dados globais.")


def render_nova_analise(user_id, model, backend, class_names,
                        frame_skip, input_size, conf, nms, limiar_parada_px,
                        num_placas=1, ampliar_areas=False, gerar_video=False,
                        coletar_candidatos=False, limiar_revisao=0.5):
    st.session_state.setdefault("pixels_per_mm", 0.0)
    uploaded = st.file_uploader(
        "Escolha um arquivo de vídeo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"]
    )
    if uploaded is None:
        st.info("Aguardando o upload do vídeo...")
        return

    # Salva o upload em disco por streaming, uma única vez por arquivo
    token = (uploaded.name, uploaded.size)
    if st.session_state.get("video_token") != token:
        caminho_antigo = st.session_state.pop("video_path", None)
        if caminho_antigo and os.path.exists(caminho_antigo):
            try:
                os.remove(caminho_antigo)
            except OSError:
                pass
        sufixo = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, tmp, length=8 * 1024 * 1024)
        st.session_state["video_path"] = tmp.name
        st.session_state["video_token"] = token
        st.session_state.pop("resultado", None)
        log_event("video_uploaded", f"Upload de '{uploaded.name}'",
                  user_id=user_id, tamanho_bytes=uploaded.size)

    video_path = st.session_state["video_path"]
    # Altura/centralização do vídeo agora vêm do CSS global (ui.py).
    st.video(video_path)

    # Trecho do vídeo a analisar (início e fim em hh:mm:ss)
    dur = _duracao_video(video_path)
    if dur and dur >= 1:
        t_min, t_max = dtime(0, 0, 0), _seg_para_time(int(dur))
        ini_t, fim_t = st.slider(
            "Trecho a analisar (hh:mm:ss)", min_value=t_min, max_value=t_max,
            value=(t_min, t_max), step=dtimedelta(seconds=1), format="HH:mm:ss",
            help="Defina o início e o fim. Só esse trecho do vídeo será "
                 "processado — útil para ignorar manuseio no começo/fim.",
        )
        inicio_s = float(_time_para_seg(ini_t))
        # Mantendo o fim no máximo, processa até o fim real do vídeo.
        fim_s = None if fim_t == t_max else float(_time_para_seg(fim_t))
        fim_efetivo = dur if fim_s is None else fim_s
        if fim_efetivo <= inicio_s:
            st.warning("O fim deve ser maior que o início.")
        st.caption(f"Analisando de **{_fmt_hms(inicio_s)}** a "
                   f"**{_fmt_hms(fim_efetivo)}** (de {_fmt_hms(dur)} totais).")
    else:
        inicio_s, fim_s = 0.0, None

    # Ferramentas de desenho (escala/áreas): apenas UMA fica ativa por vez.
    # Dois st_canvas na mesma página conflitam e às vezes não renderizam.
    st.markdown("#### Preparação (opcional)")
    escala_atual = st.session_state.get("pixels_per_mm", 0.0)
    zonas_atuais = st.session_state.get("zones") or []
    st.caption(
        f"Escala: **{('%.2f px/mm' % escala_atual) if escala_atual > 0 else '—'}** · "
        f"Áreas definidas: **{len(zonas_atuais)}**"
    )
    ferramenta = st.radio(
        "Ferramenta de desenho",
        ["Nenhuma", "📏 Calibrar escala (px → mm)", "🎯 Definir áreas de monitoramento"],
        horizontal=True, key="ferramenta_desenho",
        help="Só uma ferramenta de desenho fica ativa por vez — isso evita o "
             "conflito que fazia o canvas não abrir em algumas vezes.",
    )
    if ferramenta.startswith("📏"):
        ferramenta_calibracao(video_path)
    elif ferramenta.startswith("🎯"):
        ferramenta_zonas(video_path)

    if st.button("Processar Vídeo", type="primary"):
        zonas = st.session_state.get("zones") or []
        params = dict(frame_skip=frame_skip, input_size=input_size,
                      conf=conf, nms=nms, backend=backend,
                      pixels_per_mm=st.session_state.get("pixels_per_mm", 0.0) or None,
                      zones=json.dumps(zonas) if zonas else None)
        # Com áreas definidas, cada área é uma placa: o nº de abelhas = nº de áreas.
        n_alvos = len(zonas) if zonas else num_placas
        usar_ampliacao = bool(zonas) and ampliar_areas
        if zonas:
            msg = (f"{len(zonas)} área(s) definida(s): cada área será tratada como "
                   "uma placa. Detecções fora das áreas são descartadas.")
            if usar_ampliacao:
                msg += " Detecção ampliada por área **ativada**."
            st.info(msg)
        elif ampliar_areas:
            st.warning("Detecção ampliada por área ignorada: nenhuma área definida.")
        log_params = {k: v for k, v in params.items() if k != "zones"}
        log_event("analysis_started", f"Iniciando análise de '{uploaded.name}'",
                  user_id=user_id, num_placas=n_alvos, usa_zonas=bool(zonas),
                  **log_params)
        try:
            with st.spinner(f"Processando vídeo (1 a cada {frame_skip} frames, "
                            f"{n_alvos} placa(s))..."):
                resultado = processar_video(
                    video_path, model, class_names, frame_skip, conf, nms,
                    num_alvos=num_placas, zones=zonas or None,
                    ampliar_areas=usar_ampliacao, gerar_video=gerar_video,
                    inicio_s=inicio_s, fim_s=fim_s,
                    coletar_candidatos=coletar_candidatos, limiar_revisao=limiar_revisao,
                )
        except Exception as e:
            log_event("analysis_failed", f"Erro ao processar '{uploaded.name}': {e}",
                      user_id=user_id, level=logging.ERROR)
            st.error(f"Erro durante o processamento: {e}")
            return

        if resultado is None or resultado.primeiro_frame is None:
            st.error("Não foi possível ler nenhum frame do vídeo.")
            return

        analysis_id = salvar_analise(
            user_id, uploaded.name, uploaded.size, params, resultado
        )
        st.session_state["resultado"] = resultado
        st.session_state["resultado_frame_skip"] = frame_skip
        st.session_state["resultado_pixels_per_mm"] = params["pixels_per_mm"]
        st.session_state["resultado_zones"] = zonas
        # Vídeo de conferência: remove o anterior e guarda o novo caminho.
        antigo_video = st.session_state.pop("video_deteccoes", None)
        if antigo_video and os.path.exists(antigo_video):
            try:
                os.remove(antigo_video)
            except OSError:
                pass
        st.session_state["video_deteccoes"] = resultado.video_deteccoes
        st.session_state["video_deteccoes_nome"] = uploaded.name
        msg_extra = (" Vídeo de conferência disponível na aba '🎬 Conferência'."
                     if resultado.video_deteccoes else "")
        # Salva os candidatos para reanotação (se coletados).
        n_cand = 0
        for cand in (resultado.candidatos or []):
            database.add_candidate(
                user_id, analysis_id, uploaded.name, cand["frame_index"],
                cand["reason"], cand["pred_conf"], cand["pred_boxes"],
                cand["width"], cand["height"], cand["image_jpg"],
                area=cand.get("area"))
            n_cand += 1
        if n_cand:
            msg_extra += (f" {n_cand} candidato(s) coletado(s) para reanotação "
                          "na aba '🏷️ Anotação'.")
            log_event("candidates_collected", f"{n_cand} candidatos de '{uploaded.name}'",
                      user_id=user_id, analysis_id=analysis_id)
        st.success(f"Processamento concluído! Análise salva como **#{analysis_id}** "
                   "— disponível na aba 'Minhas Análises'." + msg_extra)

    resultado = st.session_state.get("resultado")
    if resultado is not None and resultado.primeiro_frame is not None:
        exibir_resultados(
            resultado, st.session_state.get("resultado_frame_skip", frame_skip),
            limiar_parada_px,
            pixels_per_mm=st.session_state.get("resultado_pixels_per_mm"),
            zones=st.session_state.get("resultado_zones"),
        )


def render_conferencia():
    """Aba de conferência: reproduz o vídeo anotado das detecções do último
    processamento, para o usuário verificar e ajustar os parâmetros."""
    st.markdown("### 🎬 Conferência das detecções")
    st.caption("Vídeo montado a partir dos frames processados, com as detecções "
               "desenhadas — útil para verificar pontos importantes e ajustar os "
               "parâmetros (confiança, NMS, áreas, ampliação).")
    path = st.session_state.get("video_deteccoes")
    if not path or not os.path.exists(path):
        st.info("Nenhum vídeo de conferência disponível ainda. Marque **'Gerar "
                "vídeo de conferência das detecções'** na barra lateral (seção "
                "*Detecção*) e processe um vídeo na aba *Nova Análise*.")
        return
    nome = st.session_state.get("video_deteccoes_nome")
    if nome:
        st.caption(f"Origem: **{nome}**")
    st.markdown("**Legenda:** 🟧 áreas · 🟨 detecções brutas do YOLO (com confiança) "
                "· 🟩 detecção aceita / rastreada")
    st.video(path)
    try:
        with open(path, "rb") as fh:
            st.download_button("⬇️ Baixar vídeo de conferência", data=fh.read(),
                               file_name="conferencia_deteccoes.mp4", mime="video/mp4")
    except OSError:
        st.warning("Não foi possível ler o arquivo de vídeo de conferência.")


def _box_para_rect_canvas(box, escala):
    """Caixa [x,y,w,h] (px imagem) -> objeto fabric rect (coords do canvas)."""
    x, y, w, h = box[:4]
    return {"type": "rect", "left": x / escala, "top": y / escala,
            "width": w / escala, "height": h / escala,
            "originX": "left", "originY": "top", "angle": 0,
            "scaleX": 1, "scaleY": 1, "strokeWidth": 2,
            "stroke": "#00E0A4", "fill": "rgba(0,224,164,0.15)"}


def _ultralytics_disponivel():
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


# --- Dataset externo importado (imagens originais do YOLOv4) ---

def _externo_img_dir():
    return _EXTERNAL_DIR / "images" / "train"


def _externo_lbl_dir():
    return _EXTERNAL_DIR / "labels" / "train"


def contar_externo():
    """(n_imagens, n_labels) do dataset externo importado."""
    try:
        n_i = sum(1 for p in _externo_img_dir().glob("*") if p.suffix.lower() in _IMG_EXTS)
        n_l = sum(1 for _ in _externo_lbl_dir().glob("*.txt"))
        return n_i, n_l
    except Exception:
        return 0, 0


def importar_dataset_zip(conteudo_bytes):
    """Importa um .zip no formato YOLO (imagens + labels) para o dataset externo.

    Aceita qualquer layout: separa por extensão (imagens vs .txt) pelo nome do
    arquivo. Mescla com o que já foi importado. Devolve (n_imgs, n_labels) novos.
    """
    import io as _io
    import zipfile
    img_dir, lbl_dir = _externo_img_dir(), _externo_lbl_dir()
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_i = n_l = 0
    with zipfile.ZipFile(_io.BytesIO(conteudo_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            nome = os.path.basename(info.filename)
            if not nome or nome.startswith("."):
                continue
            ext = os.path.splitext(nome)[1].lower()
            if ext in _IMG_EXTS:
                with zf.open(info) as src, open(img_dir / nome, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n_i += 1
            elif ext == ".txt" and nome.lower() not in ("classes.txt", "readme.txt"):
                with zf.open(info) as src, open(lbl_dir / nome, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n_l += 1
    return n_i, n_l


def limpar_externo():
    """Remove todo o dataset externo importado."""
    if _EXTERNAL_DIR.exists():
        shutil.rmtree(_EXTERNAL_DIR, ignore_errors=True)


# --- Avaliação/comparação de modelos (unificada Darknet + Ultralytics) ---

def _iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def itens_avaliacao(user_id, limite=80):
    """Conjunto rotulado p/ comparar modelos: (imagem_bgr, [caixas GT xywh px]).

    Junta os candidatos anotados (e 'sem abelha' = negativas) com o dataset
    externo importado. Amostra até `limite` itens.
    """
    itens = []
    for c in database.list_annotated_candidates(user_id):
        if not c.get("image_jpg"):
            continue
        img = cv2.imdecode(np.frombuffer(c["image_jpg"], np.uint8), cv2.IMREAD_COLOR)
        gt = json.loads(c["labels"]) if c.get("labels") else []
        itens.append((img, [b[:4] for b in gt]))
    img_dir, lbl_dir = _externo_img_dir(), _externo_lbl_dir()
    if img_dir.is_dir():
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() not in _IMG_EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            gt = []
            lp = lbl_dir / (p.stem + ".txt")
            if lp.exists():
                for ln in lp.read_text(errors="ignore").splitlines():
                    parts = ln.split()
                    if len(parts) >= 5:
                        cx, cy, w, h = (float(v) for v in parts[1:5])
                        gt.append([(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H])
            itens.append((img, gt))
    if len(itens) > limite:
        idx = np.linspace(0, len(itens) - 1, limite).astype(int)
        itens = [itens[i] for i in idx]
    return itens


def avaliar_modelo(model, itens, conf, nms, id_alvo, iou_thr=0.5):
    """Roda o modelo nas imagens rotuladas e calcula P/R/F1 e IoU médio.

    Casamento guloso por IoU (≥ iou_thr) entre detecções e ground-truth. Vale
    para Darknet e Ultralytics (ambos expõem .detect(frame, conf, nms))."""
    tp = fp = fn = 0
    ious = []
    for img, gts in itens:
        cl, sc, bx = model.detect(img, conf, nms)
        dets = []
        if len(sc) > 0:
            sc = np.asarray(sc).reshape(-1)
            cl = np.asarray(cl).reshape(-1)
            bx = np.asarray(bx).reshape(-1, 4)
            if id_alvo >= 0 and len(cl) == len(sc):
                m = cl == id_alvo
                sc, bx = sc[m], bx[m]
            order = np.argsort(-sc)
            dets = [bx[i].tolist() for i in order]
        usados = set()
        for d in dets:
            melhor_i, melhor_iou = -1, 0.0
            for gi, g in enumerate(gts):
                if gi in usados:
                    continue
                v = _iou_xywh(d, g)
                if v > melhor_iou:
                    melhor_iou, melhor_i = v, gi
            if melhor_i >= 0 and melhor_iou >= iou_thr:
                tp += 1
                usados.add(melhor_i)
                ious.append(melhor_iou)
            else:
                fp += 1
        fn += len(gts) - len(usados)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precisao": prec, "recall": rec,
            "f1": f1, "iou_medio": float(np.mean(ious)) if ious else 0.0}


def importar_modelo_upload(arquivos):
    """Salva modelos enviados (.pt, .weights, .cfg) em _TRAINED_DIR e valida.

    `.pt` (Ultralytics) e `.weights`+`.cfg` (Darknet) treinados fora da
    plataforma. Devolve (salvos: list[str], erros: list[str]). Um `.pt` inválido
    é removido; um `.weights` sem `.cfg` compatível é mantido com aviso (o usuário
    pode enviar o `.cfg` depois).
    """
    _TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    salvos, erros, gravados = [], [], []
    for f in arquivos:
        nome = os.path.basename(f.name)
        ext = os.path.splitext(nome)[1].lower()
        if ext not in (".pt", ".weights", ".cfg"):
            erros.append(f"{nome}: extensão não suportada (use .pt, .weights ou .cfg).")
            continue
        destino = _TRAINED_DIR / nome
        with open(destino, "wb") as out:
            out.write(f.getbuffer())
        gravados.append(destino)

    for p in gravados:
        ext = p.suffix.lower()
        if ext == ".cfg":
            continue  # validado junto com o .weights correspondente
        if ext == ".pt":
            try:
                from ultralytics import YOLO
                YOLO(str(p))
                salvos.append(p.name)
            except Exception as e:
                p.unlink(missing_ok=True)
                msg = (str(e).strip().splitlines() or [repr(e)])[-1][:90]
                erros.append(f"{p.name}: .pt inválido ({msg}). Removido.")
        elif ext == ".weights":
            cfg_par = p.with_suffix(".cfg")
            cfg = str(cfg_par) if cfg_par.exists() else YOLO_CFG
            try:
                cv2.dnn.readNetFromDarknet(cfg, str(p))
                salvos.append(f"{p.name} (cfg: {os.path.basename(cfg)})")
            except Exception as e:
                msg = (str(e).strip().splitlines() or [repr(e)])[-1][:90]
                if not cfg_par.exists():
                    erros.append(f"{p.name}: enviado, mas precisa do .cfg "
                                 f"correspondente (envie {p.stem}.cfg).")
                    salvos.append(p.name)  # mantém; falta só o cfg
                else:
                    p.unlink(missing_ok=True)
                    erros.append(f"{p.name}: Darknet inválido ({msg}). Removido.")
    return salvos, erros


def treinar_modelo(user_id, base_model, epochs, imgsz, barra, status):
    """Fine-tuning Ultralytics a partir dos candidatos anotados do usuário.

    Monta o dataset YOLO em disco (candidatos + dataset externo importado),
    treina (GPU se disponível) e salva o melhor `.pt` em _TRAINED_DIR (volume de
    dados, persistente). Devolve (caminho_pt, n_imagens, n_caixas).
    """
    from ultralytics import YOLO
    import torch

    anotados = database.list_annotated_candidates(user_id)
    tmp = tempfile.mkdtemp(prefix="bee_ds_")
    extra = [str(_externo_img_dir())] if contar_externo()[0] > 0 else None
    data_yaml, n_imgs, n_caixas = dataset.export_to_dir(anotados, tmp,
                                                        extra_train_dirs=extra)

    device = 0 if torch.cuda.is_available() else "cpu"
    _TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    proj = str(_TRAINED_DIR / "_runs")
    nome_run = datetime.now().strftime("ft_%Y%m%d_%H%M%S")

    modelo = YOLO(base_model)

    def _cb(trainer):
        ep = int(getattr(trainer, "epoch", 0)) + 1
        try:
            barra.progress(min(ep / max(int(epochs), 1), 1.0))
            status.text(f"Época {ep}/{epochs}…")
        except Exception:
            pass

    modelo.add_callback("on_train_epoch_end", _cb)
    # workers=0: sem subprocessos de DataLoader → não depende do /dev/shm do
    # container (que por padrão tem só 64MB e estoura o treino).
    modelo.train(data=data_yaml, epochs=int(epochs), imgsz=int(imgsz),
                 device=device, project=proj, name=nome_run, exist_ok=True,
                 verbose=False, plots=False, workers=0)

    best = Path(proj) / nome_run / "weights" / "best.pt"
    destino = _TRAINED_DIR / f"beemove_{nome_run}.pt"
    if best.exists():
        shutil.copy(str(best), str(destino))
    return destino, n_imgs, n_caixas


def _rect_para_box(o, escala):
    """Objeto fabric rect -> caixa [x,y,w,h] em px da imagem (honra origin)."""
    w = float(o.get("width", 0)) * float(o.get("scaleX", 1))
    h = float(o.get("height", 0)) * float(o.get("scaleY", 1))
    left, top = float(o.get("left", 0)), float(o.get("top", 0))
    ox, oy = o.get("originX", "left"), o.get("originY", "top")
    x = left - (w if ox == "right" else w / 2 if ox == "center" else 0)
    y = top - (h if oy == "bottom" else h / 2 if oy == "center" else 0)
    return [round(x * escala, 1), round(y * escala, 1),
            round(w * escala, 1), round(h * escala, 1)]


def render_anotacao(user_id):
    """Reanotação de candidatos (abelhas mal/não detectadas) + export do dataset YOLO."""
    ui.header("Anotação para fine-tuning",
              "Corrija as detecções fracas e gere um dataset YOLO")
    cont = database.candidate_counts(user_id)
    pend = cont.get("pendente", 0)
    anot = cont.get("anotado", 0) + cont.get("sem_abelha", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pendentes", pend)
    c2.metric("Anotados", anot)
    c3.metric("Descartados", cont.get("descartado", 0))

    if pend + anot == 0:
        st.info("Nenhum candidato coletado ainda. Marque **'🏷️ Coletar candidatos "
                "para reanotação'** na barra lateral e processe um vídeo — ou "
                "importe um dataset externo abaixo e treine direto.")

    # --- Importar dataset externo (imagens originais do YOLOv4) ---
    with st.expander("📥 Importar imagens externas (dataset YOLO)", expanded=False):
        n_ext_i, n_ext_l = contar_externo()
        st.caption(f"Dataset externo atual: **{n_ext_i} imagem(ns)**, {n_ext_l} "
                   "arquivo(s) de rótulo. Importe aqui as imagens originais usadas "
                   "no YOLOv4 (formato YOLO: imagens + .txt). Elas entram no treino "
                   "junto com as reanotadas.")
        zf = st.file_uploader("Arquivo .zip (imagens + labels YOLO)", type=["zip"],
                              key="ext_zip")
        if zf is not None and st.button("Importar .zip"):
            try:
                ni, nl = importar_dataset_zip(zf.getvalue())
                log_event("dataset_imported", f"Dataset externo: +{ni} imgs, +{nl} labels",
                          user_id=user_id)
                st.success(f"Importado: +{ni} imagem(ns), +{nl} rótulo(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao importar: {e}")
        if n_ext_i > 0 and st.button("🗑️ Limpar dataset externo"):
            limpar_externo()
            st.rerun()

    # --- Exportar dataset YOLO ---
    with st.expander("📦 Exportar dataset YOLO", expanded=False):
        anotados = database.list_annotated_candidates(user_id)
        res = dataset.resumo(anotados)
        st.caption(f"{res['imagens']} imagem(ns) · {res['caixas']} caixa(s) · "
                   f"{res['negativas']} negativa(s) (sem abelha). Inclui apenas "
                   "candidatos com status *anotado* ou *sem abelha*.")
        if res["imagens"] > 0 and st.button("Gerar dataset (.zip)"):
            st.session_state["dataset_zip"] = dataset.build_yolo_zip(anotados)
            log_event("dataset_exported",
                      f"Dataset YOLO ({res['imagens']} imgs, {res['caixas']} caixas)",
                      user_id=user_id)
        if st.session_state.get("dataset_zip"):
            st.download_button("⬇️ Baixar dataset YOLO", data=st.session_state["dataset_zip"],
                               file_name="dataset_yolo_beetracker.zip",
                               mime="application/zip")

    # --- Treinar modelo (fine-tuning Ultralytics) ---
    with st.expander("🎓 Treinar modelo (fine-tuning)", expanded=False):
        if not _ultralytics_disponivel():
            st.warning("Ultralytics não está instalado nesta imagem. Reconstrua a "
                       "imagem (o `ultralytics` foi adicionado ao requirements) para "
                       "habilitar o treino dentro do app.")
        else:
            anotados = database.list_annotated_candidates(user_id)
            res = dataset.resumo(anotados)
            n_ext_i, _ = contar_externo()
            total_imgs = res["imagens"] + n_ext_i
            st.caption(f"Treina um modelo YOLO (Ultralytics) com **{res['imagens']} "
                       f"reanotada(s)** + **{n_ext_i} externa(s)** = {total_imgs} "
                       "imagem(ns). O modelo `.pt` resultante fica salvo e pode ser "
                       "selecionado/comparado na **Administração**.")
            base = st.selectbox("Modelo base", ["yolov8n.pt", "yolov8s.pt",
                                                "yolo11n.pt", "yolo11s.pt"],
                                help="Modelos leves treinam mais rápido. Baixados "
                                     "automaticamente na 1ª vez (requer internet).")
            cep, cimg = st.columns(2)
            epochs = cep.slider("Épocas", 10, 300, 100, 10)
            imgsz = cimg.select_slider("imgsz (resolução)", [320, 416, 512, 640],
                                       value=640)
            if total_imgs < 5:
                st.info("Reúna ao menos ~5 imagens (reanotadas e/ou importadas; "
                        "idealmente dezenas) para um treino mínimo viável.")
            elif st.button("🚀 Treinar agora", type="primary"):
                barra = st.progress(0.0)
                status = st.empty()
                try:
                    with st.spinner("Treinando — pode levar vários minutos…"):
                        destino, ni, nc = treinar_modelo(
                            user_id, base, epochs, imgsz, barra, status)
                    log_event("model_trained",
                              f"Treino Ultralytics: {destino.name} ({ni} imgs, {nc} caixas)",
                              user_id=user_id)
                    if destino.exists():
                        st.success(f"✅ Treino concluído! Modelo salvo como "
                                   f"**{destino.name}**. Ative-o na aba "
                                   "**⚙️ Administração → 🧠 Modelo de detecção** "
                                   "para comparar os resultados.")
                    else:
                        st.error("Treino terminou mas o arquivo do modelo não foi "
                                 "encontrado. Verifique os logs.")
                except Exception as e:
                    st.error(f"Falha no treino: {e}")
                    log_event("model_train_failed", f"Erro no treino: {e}",
                              user_id=user_id, level=logging.ERROR)
                finally:
                    barra.empty()
                    status.empty()

    # --- Seleção do candidato a reanotar ---
    if pend + anot == 0:
        return  # sem candidatos: só import/export/treino acima
    st.markdown("### ✍️ Reanotar candidatos")
    try:
        _patch_canvas_compat()
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception:
        st.warning("Componente de desenho indisponível nesta instalação.")
        return

    filtro = st.radio("Mostrar", ["Pendentes", "Todos"], horizontal=True,
                      key="anot_filtro")
    lista = database.list_candidates(user_id,
                                     status="pendente" if filtro == "Pendentes" else None)
    if not lista:
        st.success("Nenhum candidato pendente — tudo anotado! 🎉")
        return

    def _rotulo(c):
        area = f" · {c['area']}" if c.get("area") else ""
        return (f"#{c['id']}{area} · frame {c['frame_index']} · {c['reason']} "
                f"· conf {c['pred_conf']:.2f} · [{c['status']}]")

    opcoes = {_rotulo(c): c["id"] for c in lista}
    escolhido = st.selectbox("Candidato", list(opcoes.keys()), key="anot_sel")
    cand = database.get_candidate(opcoes[escolhido], user_id)
    if not cand or not cand.get("image_jpg"):
        st.warning("Candidato sem imagem.")
        return

    frame = cv2.imdecode(np.frombuffer(cand["image_jpg"], np.uint8), cv2.IMREAD_COLOR)
    altura, largura = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Recortes de área são pequenos — amplia o canvas (mín. 500px) p/ anotar bem;
    # frames inteiros são limitados a 760px. escala converte canvas -> imagem.
    canvas_w = max(1, min(760, max(largura, 500)))
    canvas_h = int(altura * canvas_w / largura)
    escala = largura / canvas_w

    # Pré-carrega caixas: labels já anotados, senão as sugestões da rede.
    base = json.loads(cand["labels"]) if cand.get("labels") else None
    if not base:
        base = [b[:4] for b in (json.loads(cand["pred_boxes"]) if cand.get("pred_boxes") else [])]
    init = {"version": "4.4.0",
            "objects": [_box_para_rect_canvas(b, escala) for b in (base or [])]}

    st.markdown("Desenhe um **retângulo por abelha** (clique e arraste). Selecione "
                "uma caixa para mover/redimensionar; use a lixeira da barra do canvas "
                "para excluir. Se não houver abelha, use *Sem abelha*.")
    col_canvas, col_acoes = st.columns([3, 1])
    with col_canvas:
        canvas = st_canvas(
            fill_color="rgba(0,224,164,0.15)", stroke_width=2, stroke_color="#00E0A4",
            background_image=pil, update_streamlit=True, initial_drawing=init,
            height=canvas_h, width=canvas_w, drawing_mode="rect",
            key=f"canvas_anot_{cand['id']}",
        )
    with col_acoes:
        st.caption(f"Frame {cand['frame_index']} · {largura}×{altura}px")
        if st.button("💾 Salvar anotação", type="primary", width="stretch"):
            objs = (canvas.json_data or {}).get("objects") if canvas else None
            caixas = [_rect_para_box(o, escala) for o in (objs or [])
                      if o.get("type") in ("rect", "Rect") and o.get("width")]
            caixas = [b for b in caixas if b[2] > 1 and b[3] > 1]
            if not caixas:
                st.warning("Nenhuma caixa desenhada. Use *Sem abelha* se for o caso.")
            else:
                database.set_candidate_labels(cand["id"], user_id, caixas, "anotado")
                st.success(f"{len(caixas)} caixa(s) salvas.")
                st.rerun()
        if st.button("🚫 Sem abelha (negativa)", width="stretch"):
            database.set_candidate_labels(cand["id"], user_id, [], "sem_abelha")
            st.rerun()
        if st.button("🗑️ Descartar", width="stretch"):
            database.set_candidate_labels(cand["id"], user_id, [], "descartado")
            st.rerun()


def render_admin(current_user_id, input_size, usar_gpu, fp16, nms, class_names):
    """Painel de administração: usuários, modelo ativo e comparação de modelos."""
    ui.header("Administração", "Gestão de usuários e modelos")
    usuarios = database.list_users()

    df = pd.DataFrame([{
        "ID": u["id"],
        "Usuário": u["username"],
        "Perfil": "Administrador" if u["role"] == "admin" else "Comum",
        "Análises": u["n_analises"],
        "Criado em": (u["created_at"] or "")[:16],
        "Último acesso": (u["last_login_at"] or "—")[:16],
    } for u in usuarios])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"{len(usuarios)} usuário(s) · {database.count_admins()} administrador(es). "
               "Excluir um usuário remove também todas as análises dele.")

    col_novo, col_gerir = st.columns(2)
    with col_novo:
        st.markdown("#### ➕ Novo usuário")
        with st.form("admin_novo_user", clear_on_submit=True):
            nu = st.text_input("Usuário")
            ns = st.text_input("Senha", type="password")
            nr = st.selectbox("Perfil", ["Comum", "Administrador"])
            if st.form_submit_button("Criar usuário", type="primary",
                                     width="stretch"):
                ok, msg = auth.admin_create_user(
                    nu, ns, "admin" if nr == "Administrador" else "user")
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with col_gerir:
        st.markdown("#### 🛠️ Gerenciar usuário")
        opcoes = {f"#{u['id']} — {u['username']}": u for u in usuarios}
        sel = st.selectbox("Selecione o usuário", list(opcoes.keys()),
                           key="admin_sel_user")
        alvo = opcoes[sel]
        eh_proprio = alvo["id"] == current_user_id

        novo_papel = "user" if alvo["role"] == "admin" else "admin"
        rotulo_papel = ("Rebaixar para Comum" if alvo["role"] == "admin"
                        else "Promover a Administrador")
        c1, c2 = st.columns(2)
        if c1.button(rotulo_papel, width="stretch", key="admin_toggle_role"):
            ok, msg = auth.admin_set_role(alvo["id"], novo_papel)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        if c2.button("🗑️ Excluir", width="stretch", key="admin_del_user",
                     disabled=eh_proprio,
                     help="Você não pode excluir a própria conta." if eh_proprio else None):
            ok, msg = auth.admin_delete_user(alvo["id"])
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        with st.form("admin_reset_pwd", clear_on_submit=True):
            nova = st.text_input("Redefinir senha do usuário", type="password")
            if st.form_submit_button("Redefinir senha", width="stretch"):
                ok, msg = auth.admin_reset_password(alvo["id"], nova)
                (st.success if ok else st.error)(msg)

    # --- Subir modelo pré-treinado (fora da plataforma) ---
    st.markdown("---")
    st.markdown("#### 📤 Subir modelo pré-treinado")
    st.caption("Envie um modelo treinado fora daqui: **.pt** (Ultralytics) ou "
               "**.weights + .cfg** (Darknet/YOLOv4). Ele passa a aparecer na "
               "seleção e na comparação abaixo. Modelos ficam no volume de dados.")
    ups = st.file_uploader("Arquivos do modelo (.pt, ou .weights + .cfg)",
                           type=["pt", "weights", "cfg"], accept_multiple_files=True,
                           key="admin_upload_modelo")
    if ups and st.button("Enviar modelo(s)", key="admin_enviar_modelo"):
        try:
            salvos, erros = importar_modelo_upload(ups)
            if salvos:
                log_event("model_uploaded", f"Upload de modelo: {', '.join(salvos)}",
                          user_id=current_user_id)
                st.success("Salvo: " + " · ".join(salvos))
            for e in erros:
                st.warning(e)
        except Exception as e:
            st.error(f"Falha no upload: {e}")

    # --- Modelo de detecção ativo ---
    st.markdown("---")
    st.markdown("#### 🧠 Modelo de detecção")
    modelos = listar_modelos()
    _, weights_atual = modelo_ativo()
    if not modelos:
        st.info(f"Nenhum modelo encontrado em `{_MODELS_DIR}` ou `{_TRAINED_DIR}`. "
                "Coloque os pesos lá ou treine um na aba *🏷️ Anotação*.")
    else:
        rotulos = {f"{m['nome']}  ·  {m['tipo']}": m for m in modelos}
        atual_nome = os.path.basename(weights_atual)
        chaves = list(rotulos.keys())
        idx = next((i for i, k in enumerate(chaves)
                    if rotulos[k]["nome"] == atual_nome), 0)
        sel_rot = st.selectbox("Pesos em uso", chaves, index=idx, key="admin_modelo")
        st.caption(f"Ativo agora: **{atual_nome}**. Trocar recarrega o modelo "
                   "para você comparar os resultados (Darknet = modelo original; "
                   "Ultralytics = modelos `.pt` treinados no app).")
        if st.button("Aplicar modelo", type="primary", key="admin_aplicar_modelo"):
            m = rotulos[sel_rot]
            sel = m["nome"]
            database.set_setting("active_weights", m["weights"])
            database.set_setting("active_cfg", m["cfg"])
            carregar_modelo.clear()   # invalida o cache p/ recarregar os novos pesos
            log_event("model_selected", f"Modelo ativo: {sel}",
                      user_id=current_user_id)
            st.success(f"Modelo **{sel}** ativado. Recarregando…")
            st.rerun()

    # --- Comparar modelos (em conjunto rotulado) ---
    st.markdown("---")
    st.markdown("#### 🆚 Comparar modelos")
    itens = itens_avaliacao(current_user_id)
    if not itens:
        st.info("Para comparar, é preciso um conjunto rotulado: anote candidatos "
                "(aba *🏷️ Anotação*) e/ou importe um dataset externo com rótulos.")
    elif not modelos:
        st.info("Nenhum modelo disponível para comparar.")
    else:
        st.caption(f"Avalia cada modelo em **{len(itens)} imagem(ns) rotulada(s)** "
                   "(reanotadas + externas), casando detecções ao ground-truth por "
                   "IoU ≥ 0,5. Maior **F1** = melhor equilíbrio entre acertar e não "
                   "inventar. Útil para escolher qual ativar acima.")
        conf_aval = st.slider("Confiança da avaliação", 0.1, 0.9, 0.4, 0.05,
                              key="admin_aval_conf")
        if st.button("Avaliar todos os modelos", key="admin_avaliar"):
            id_alvo = (class_names.index(TARGET_CLASS)
                       if class_names and TARGET_CLASS in class_names else 0)
            linhas = []
            prog = st.progress(0.0)
            for i, m in enumerate(modelos):
                try:
                    mdl, _, _ = carregar_modelo(input_size, usar_gpu, fp16,
                                                m["cfg"], m["weights"])
                    met = avaliar_modelo(mdl, itens, conf_aval, nms, id_alvo)
                    linhas.append({
                        "Modelo": m["nome"], "Tipo": m["tipo"],
                        "Precisão": round(met["precisao"], 3),
                        "Recall": round(met["recall"], 3),
                        "F1": round(met["f1"], 3),
                        "IoU médio": round(met["iou_medio"], 3),
                        "TP": met["tp"], "FP": met["fp"], "FN": met["fn"],
                    })
                except Exception as e:
                    linhas.append({"Modelo": m["nome"], "Tipo": m["tipo"],
                                   "F1": None, "Precisão": f"erro: {e}"})
                prog.progress((i + 1) / len(modelos))
            prog.empty()
            df_cmp = pd.DataFrame(linhas).sort_values(
                "F1", ascending=False, na_position="last")
            st.session_state["cmp_modelos"] = df_cmp
            log_event("models_compared", f"Comparados {len(modelos)} modelos",
                      user_id=current_user_id)
        if st.session_state.get("cmp_modelos") is not None:
            df_cmp = st.session_state["cmp_modelos"]
            st.dataframe(df_cmp, hide_index=True, width="stretch")
            validos = df_cmp.dropna(subset=["F1"]) if "F1" in df_cmp else df_cmp
            if len(validos):
                melhor = validos.iloc[0]
                st.success(f"🏆 Melhor F1: **{melhor['Modelo']}** "
                           f"(F1={melhor['F1']}). Selecione-o acima e *Aplicar modelo*.")


# --- Fluxo principal ---

if "user_id" not in st.session_state:
    tela_login()
    st.stop()

user_id = st.session_state["user_id"]
username = st.session_state["username"]

ui.inject_css()
role = st.session_state.get("role", "user")
is_admin = role == "admin"

if not (os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS)):
    st.error(
        f"Arquivos YOLO não encontrados! Certifique-se de que '{YOLO_CFG}' e "
        f"'{YOLO_WEIGHTS}' estão no diretório da aplicação."
    )
    st.stop()

with st.sidebar:
    st.markdown(f"👤 **{username}**")
    st.markdown(ui.chip("Administrador", "admin") if is_admin
                else ui.chip("Conta comum", "info"), unsafe_allow_html=True)
    with st.expander("🔒 Minha conta"):
        with st.form("form_trocar_senha", clear_on_submit=True):
            p_atual = st.text_input("Senha atual", type="password")
            p_nova = st.text_input("Nova senha", type="password")
            p_conf = st.text_input("Confirmar nova senha", type="password")
            if st.form_submit_button("Trocar senha", width="stretch"):
                if p_nova != p_conf:
                    st.error("As senhas não conferem.")
                else:
                    ok, msg = auth.change_password(user_id, p_atual, p_nova)
                    (st.success if ok else st.error)(msg)
    if st.button("Sair", width="stretch"):
        log_event("logout", f"Logout de '{username}'", user_id=user_id)
        caminho = st.session_state.pop("video_path", None)
        if caminho and os.path.exists(caminho):
            try:
                os.remove(caminho)
            except OSError:
                pass
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.header("⚙️ Configurações")
    tem_cuda = cuda_disponivel()

    with st.expander("🖥️ Hardware / aceleração", expanded=False):
        usar_gpu = st.toggle(
            "Usar GPU (CUDA)", value=tem_cuda, disabled=not tem_cuda,
            help="Disponível apenas com OpenCV compilado com CUDA (imagem Docker GPU).",
        )
        fp16 = st.toggle(
            "Precisão FP16 na GPU", value=usar_gpu, disabled=not usar_gpu,
            help="~2x mais rápido em GPUs com Tensor Cores; precisão praticamente igual.",
        )
        if not tem_cuda:
            st.caption("GPU indisponível: OpenCV sem suporte CUDA neste ambiente.")

    with st.expander("🔍 Detecção", expanded=True):
        input_size = st.select_slider(
            "Resolução de entrada da rede", options=[320, 416, 512], value=416,
            help="320 é mais rápido; 512 detecta melhor objetos pequenos.",
        )
        frame_skip = st.slider("Processar 1 a cada N frames", 1, 10, 10)
        conf = st.slider("Limiar de confiança", 0.1, 0.9, 0.4, 0.05)
        nms = st.slider("Limiar NMS", 0.1, 0.9, 0.3, 0.05)
        ampliar_areas = st.checkbox(
            "🔍 Detecção por recorte das áreas (crop)", value=True,
            help="Quando há áreas definidas: recorta cada área (com uma folga "
                 "lateral, como no recorte do resultado por área) e passa SÓ o "
                 "recorte pela rede — a abelha ocupa muito mais pixels, melhorando "
                 "a detecção em vídeos de baixa resolução. Considera apenas as "
                 "áreas delimitadas; se a abelha de uma placa adjacente aparecer "
                 "na folga, é descartada como falso positivo (o centro precisa "
                 "estar dentro desta área e fora das vizinhas). Sem áreas "
                 "definidas, não tem efeito. Mais lento (uma detecção por área "
                 "por frame) e requer reprocessar o vídeo.",
        )
        gerar_video = st.checkbox(
            "Gerar vídeo de conferência das detecções", value=True,
            help="Cria um MP4 com as detecções desenhadas nos frames processados "
                 "(🟧 áreas · 🟨 detecções brutas do YOLO com confiança · 🟩 detecção "
                 "aceita/rastreada), exibido na aba '🎬 Conferência'. Ajuda a ajustar "
                 "os parâmetros. Aumenta um pouco o tempo de processamento.",
        )
        coletar_candidatos = st.checkbox(
            "🏷️ Coletar candidatos para reanotação", value=False,
            help="Durante o processamento, amostra frames cuja melhor detecção "
                 "fique fraca (abaixo do limiar de revisão) ou ausente e guarda-os "
                 "para reanotar na aba '🏷️ Anotação' — base do dataset de "
                 "fine-tuning.",
        )
        limiar_revisao = st.slider(
            "Limiar de revisão (confiança)", 0.1, 0.9, 0.5, 0.05,
            disabled=not coletar_candidatos,
            help="Frames cuja melhor detecção fique ABAIXO deste valor (ou sem "
                 "detecção) viram candidatos para reanotação.",
        )

    with st.expander("📊 Rastreamento e áreas", expanded=True):
        zonas_def = st.session_state.get("zones") or []
        if zonas_def:
            # Com áreas definidas, cada área é uma placa: o nº é determinado por elas
            # (um único valor manda). Não faz sentido pedir o número manualmente.
            num_placas = len(zonas_def)
            st.info(f"🎯 **{num_placas} área(s) definida(s)**: cada área é uma placa, "
                    "então o número de placas vem das áreas. Limpe as áreas (aba *Nova "
                    "Análise*) para definir o número manualmente.")
        else:
            num_placas = st.number_input(
                "Número de placas (abelhas por frame)", min_value=1, max_value=12, value=1,
                help="Mantém as N detecções de maior confiança em cada frame e rastreia "
                     "cada uma como uma placa/abelha separada. Use quando há várias placas "
                     "no mesmo vídeo. Requer reprocessar o vídeo.",
            )
        limiar_parada_px = st.slider(
            "Limiar de parada (px entre amostras)", 0, 50, 3,
            help="Deslocamentos menores que isto contam como 'parada' nas métricas "
                 "de tempo em movimento. Ajuste conforme a escala do vídeo. "
                 "Não exige reprocessar — recalcula na hora.",
        )

    with st.expander("📈 Gráficos exibidos", expanded=False):
        st.caption("Escolha quais gráficos opcionais aparecem nos resultados "
                   "(efeito imediato, sem reprocessar).")
        st.checkbox("Posição X / Y ao longo do tempo", value=True, key="graf_posicao")
        st.checkbox("Velocidade instantânea", value=True, key="graf_velocidade")
        st.checkbox("Distribuição de velocidades", value=True, key="graf_histograma")

class_names = carregar_classes()
cfg_ativo, weights_ativo = modelo_ativo()
model, backend, erro_cuda = carregar_modelo(input_size, usar_gpu, fp16,
                                            cfg_ativo, weights_ativo)
st.sidebar.markdown("---")
st.sidebar.info(f"Backend em uso: **{backend}**")
st.sidebar.caption(f"🧠 Modelo: **{os.path.basename(weights_ativo)}**")
if erro_cuda:
    st.sidebar.error(f"Motivo do fallback para CPU: {erro_cuda}")
    if "CUDA_ARCH" in erro_cuda:
        st.sidebar.warning(
            "A imagem foi compilada para outra arquitetura de GPU. "
            "Reconstrua com `--build-arg CUDA_ARCH_BIN=\"<compute capability "
            "da sua placa>\"` (veja `nvidia-smi --query-gpu=compute_cap "
            "--format=csv`)."
        )
st.sidebar.info(f"Classe alvo: **'{TARGET_CLASS}'**")
st.sidebar.caption("🌗 Tema **escuro** por padrão — para alternar para o claro, "
                   "use o menu **⋮** (canto superior direito) → *Settings* → *Theme*.")

ui.header("Detecção e Rastreamento de Abelha",
          "YOLOv4 · áreas de monitoramento · múltiplas placas · relatórios")

rotulos_abas = ["📹 Nova Análise", "📊 Minhas Análises", "🎬 Conferência",
                "🏷️ Anotação"]
if is_admin:
    rotulos_abas.append("⚙️ Administração")
abas = st.tabs(rotulos_abas)
with abas[0]:
    render_nova_analise(user_id, model, backend, class_names,
                        frame_skip, input_size, conf, nms, limiar_parada_px,
                        num_placas, ampliar_areas, gerar_video,
                        coletar_candidatos, limiar_revisao)
with abas[1]:
    render_historico(user_id, username, limiar_parada_px)
with abas[2]:
    render_conferencia()
with abas[3]:
    render_anotacao(user_id)
if is_admin:
    with abas[4]:
        render_admin(user_id, input_size, usar_gpu, fp16, nms, class_names)
