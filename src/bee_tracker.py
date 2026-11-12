"""
BeeMove — detecção e rastreamento de abelhas em placas (Darknet/OpenCV DNN ou Ultralytics).

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

import copy
import html
import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import zlib
from contextlib import contextmanager
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
import conferencia
import database
import dataset
import report
import ui
from app_logging import log_event
import i18n
from i18n import t

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

st.set_page_config(page_title=t("core.page_title"), page_icon="🐝",
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


@st.cache_resource(show_spinner=t("core.loading_model"))
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


def ativar_modelo(weights, cfg, user_id, evento="model_selected"):
    """Torna (weights, cfg) o modelo ativo de todos os usuários (Admin → Modelos)."""
    database.set_setting("active_weights", str(weights))
    database.set_setting("active_cfg", str(cfg or ""))
    carregar_modelo.clear()   # invalida o cache p/ recarregar os novos pesos
    log_event(evento, f"Modelo ativo: {os.path.basename(str(weights))}",
              user_id=user_id)


class ErroCarga(str):
    """Texto cru (sem tradução) de uma falha ao carregar o modelo + o seu tipo.

    `tipo`: "cuda"  -> o backend CUDA falhou no warm-up e o modelo roda na CPU;
            "model" -> o .pt (Ultralytics) não carregou e o Darknet padrão foi
                       usado no lugar (pode estar na GPU).
    Continua sendo um str (logs, `in`, RuntimeError); a tradução é feita só na
    exibição, por texto_erro_carga().
    """

    def __new__(cls, detalhe: str, tipo: str):
        obj = super().__new__(cls, detalhe)
        obj.tipo = tipo
        return obj


def texto_erro_carga(erro) -> str:
    """Mensagem de falha de carga pronta para a interface (no idioma atual)."""
    if getattr(erro, "tipo", None) == "model":
        return t("model.ultralytics_unavailable", error=str(erro))
    return str(erro)


# Token neutro (gravado em analyses.backend): o texto é traduzido na exibição.
BACKEND_CPU_FALLBACK = "CPU (fallback)"


def rotulo_backend(backend) -> str:
    """Backend para exibição; o fallback (inclusive o texto antigo em português
    gravado em análises anteriores) aparece traduzido."""
    if backend and "fallback" in str(backend):
        return t("model.backend_cpu_fallback")
    return backend or "—"


@st.cache_resource(show_spinner=t("core.loading_model"))
def carregar_modelo(input_size: int, usar_gpu: bool, fp16: bool,
                    cfg_path: str = None, weights_path: str = None):
    """Carrega a rede Darknet (cfg/weights selecionáveis no admin).

    Retorna (DetectionModel, descrição do backend, erro_cuda) — erro_cuda é
    None quando tudo deu certo, ou um ErroCarga com a falha que causou o
    fallback: tipo "cuda" (GPU -> CPU, ex.: imagem compilada para outra
    arquitetura de GPU) ou "model" (.pt -> Darknet padrão).
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
            erro_cuda = ErroCarga(
                str(e).strip().splitlines()[-1] if str(e) else repr(e), "model")
            log_event("model_load_failed", f"Ultralytics indisponível: {erro_cuda}",
                      level=logging.ERROR)
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
        erro_cuda = ErroCarga(
            str(e).strip().splitlines()[-1] if str(e) else repr(e), "cuda")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = BACKEND_CPU_FALLBACK
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
        # Pedido de parada (ex.: a execução do Streamlit foi interrompida no
        # meio): a thread para de decodificar e libera o VideoCapture.
        self._parar = threading.Event()

    def _por_na_fila(self, item) -> bool:
        """put() que desiste se a parada for pedida (fila cheia sem consumidor)."""
        while not self._parar.is_set():
            try:
                self.fila.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def parar(self):
        """Interrompe a leitura e esvazia a fila (libera a memória dos frames)."""
        self._parar.set()

        def _esvaziar():
            try:
                while True:
                    self.fila.get_nowait()
            except queue.Empty:
                pass

        _esvaziar()
        if self.is_alive():
            self.join(timeout=2.0)
        _esvaziar()   # um put() em curso pode ter entrado após o 1º esvaziamento
        if not self.is_alive():
            self.cap.release()   # já liberado no run(); idempotente

    def run(self):
        if self.inicio_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio_frame)
        indice = self.inicio_frame
        try:
            while indice <= self.fim_frame and not self._parar.is_set():
                if (indice - self.inicio_frame) % self.frame_skip != 0:
                    if not self.cap.grab():
                        break
                    indice += 1
                    continue
                ok, frame = self.cap.read()
                if not ok:
                    break
                if not self._por_na_fila((indice, frame)):
                    break
                indice += 1
        finally:
            self._por_na_fila(None)  # sentinela de fim
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
    frames_lidos: list = None     # índices dos frames processados (em ordem)


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


def _id_classe_alvo(class_names) -> int:
    """Índice da classe alvo (TARGET_CLASS) em class_names; -1 = aceitar todas."""
    if class_names and TARGET_CLASS in class_names:
        return class_names.index(TARGET_CLASS)
    return -1


def _zonas_validas(zones):
    """(polígonos (N,2), nomes) das áreas com ≥3 pontos; ([], []) sem áreas.

    O índice na lista é o track_id da área no processamento.
    """
    polys, nomes = [], []
    for z in zones or []:
        pts = z.get("points") or []
        if len(pts) >= 3:
            polys.append(np.asarray(pts, float).reshape(-1, 2))
            nomes.append(z.get("name") or t("core.zone_default", n=len(nomes) + 1))
    return polys, nomes


def _montar_candidato(indice, img_bgr, melhor_s, sugeridas, area=None):
    """Dict de candidato a reanotação (formato de database.add_candidate) ou None.

    `melhor_s` = 0 → 'sem_deteccao'; acima de 0 → 'baixa_confianca'.
    """
    ok_jpg, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok_jpg:
        return None
    h0, w0 = img_bgr.shape[:2]
    return {
        "frame_index": int(indice),
        "reason": "sem_deteccao" if melhor_s == 0 else "baixa_confianca",
        "pred_conf": float(melhor_s), "pred_boxes": sugeridas,
        "width": int(w0), "height": int(h0),
        "image_jpg": jpg.tobytes(), "area": area,
    }


def _caixas_alvo(model, img, conf, nms, id_alvo):
    """Passe de detecção filtrado à classe alvo: [[x,y,w,h,score], ...] (px)."""
    classes, scores, boxes = model.detect(img, conf, nms)
    if len(scores) == 0:
        return []
    scores = np.asarray(scores).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    boxes = np.asarray(boxes).reshape(-1, 4)
    if id_alvo >= 0:
        m = classes == id_alvo
        scores, boxes = scores[m], boxes[m]
    return [[float(x), float(y), float(w), float(h), float(s)]
            for s, (x, y, w, h) in zip(scores, boxes)]


def processar_video(video_path, model, class_names, frame_skip, conf, nms,
                    num_alvos=1, zones=None, ampliar_areas=False, gerar_video=False,
                    inicio_s=0.0, fim_s=None, coletar_candidatos=False,
                    conf_captura=0.15, limiar_revisao=0.5, max_candidatos=200,
                    fase=None):
    """`fase`: callback opcional fase(texto) para anunciar as etapas na interface
    (ex.: dentro de um st.status). Não altera o processamento."""
    fase = fase or (lambda _msg: None)
    leitor = LeitorDeFrames(video_path, frame_skip, inicio_s, fim_s)
    if not leitor.aberto:
        leitor.cap.release()
        st.error(t("core.video_open_error"))
        return None
    fase(t("analysis.phase_reading", total=leitor.total_frames, skip=max(1, frame_skip),
           n=-(-leitor.total_frames // max(1, frame_skip))))

    id_alvo = _id_classe_alvo(class_names)
    if class_names and id_alvo < 0:
        st.warning(t("core.class_not_found", cls=TARGET_CLASS, names=YOLO_NAMES))

    # Quando há áreas definidas, cada área é uma placa/abelha: a detecção é
    # atribuída à área que a contém e detecções fora de todas as áreas são
    # descartadas como falso positivo. Sem áreas, usa o rastreador por
    # vizinho mais próximo com as N detecções de maior confiança.
    zone_paths = None
    zone_polys = None
    zone_names = []
    if zones:
        from matplotlib.path import Path as _MplPath
        zone_polys, zone_names = _zonas_validas(zones)
        if zone_polys:
            zone_paths = [_MplPath(p) for p in zone_polys]
            num_alvos = len(zone_paths)
        else:
            zone_polys = None

    total = max(leitor.total_frames, 1)
    fase(t("analysis.phase_detecting", n=num_alvos))
    barra = st.progress(0)
    status = st.empty()
    gravador = None
    leitor.start()
    concluido = False
    try:

        centroides, indices_frames, track_ids = [], [], []
        frames_lidos = []   # todos os frames processados (p/ a Conferência)
        rastreador = RastreadorMultiplo(num_alvos)
        primeiro_frame = None
        processados = 0
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
            frames_lidos.append(int(indice))
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
                            aceitos.append((cx, cy, t("core.bee_label", n=tid + 1)))

            # Candidatos para reanotação (frame/recorte LIMPO, antes de desenhar):
            # passe de baixa confiança; guarda os fracos/ausentes. Com áreas, gera
            # UM candidato por área (recorte) — imagem menor, abelha maior.
            def _guardar_candidato(img_bgr, melhor_s, sugeridas, area=None):
                cand = _montar_candidato(indice, img_bgr, melhor_s, sugeridas, area)
                if cand is not None:
                    candidatos.append(cand)

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
                    sugeridas = _caixas_alvo(model, frame, conf_captura, nms, id_alvo)
                    melhor_s = max((b[4] for b in sugeridas), default=0.0)
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
                decorrido = max(time.perf_counter() - inicio, 1e-6)
                fps_proc = processados / decorrido
                # Tempo restante: ritmo de avanço no trecho (frames do vídeo/s).
                restante = max(0, total - rel) * decorrido / max(rel, 1)
                status.text(t("core.progress_status", rel=rel, total=total,
                              fps=fps_proc, n=len(centroides),
                              eta=_fmt_duracao(restante)))

        tempo = time.perf_counter() - inicio
        barra.empty()
        status.empty()
        if gravador is not None:
            fase(t("analysis.phase_video"))
        video_conf = gravador.finalizar() if gravador is not None else None
        concluido = True
    finally:
        if not concluido:
            # Interrompido no meio (erro, ou a execução foi trocada/parada pelo
            # Streamlit — RerunException/StopException não são Exception):
            # para a thread de leitura e apaga o vídeo de conferência parcial.
            leitor.parar()
            if gravador is not None:
                gravador.descartar()

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
        frames_lidos=frames_lidos,
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

def _mostrar(fig, vazio=None):
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(vazio or t("results.no_data_chart"))


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
        st.info(t("results.no_zones"))
        return

    # Números crus no DataFrame (a ordenação funciona) e formatação/unidade no
    # cabeçalho via column_config. Uma unidade por coluna: mm com escala, senão px.
    tem_mm = all("distancia_dentro_mm" in m for m in metr_z)
    linhas = []
    for m in metr_z:
        linha = {
            "area": m["nome"],
            "detections": m["deteccoes_dentro"],
            "pct": m["perc_deteccoes"],
            "visits": m["visitas"],
            "residence": m.get("tempo_permanencia"),
        }
        if tem_mm:
            linha["dist"] = m["distancia_dentro_mm"]
            linha["area_mm2"] = m.get("area_mm2")
        else:
            linha["dist"] = m["distancia_dentro_px"]
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    if df["residence"].isna().all():
        df = df.drop(columns="residence")
    if "area_mm2" in df and df["area_mm2"].isna().all():
        df = df.drop(columns="area_mm2")
    unid = "mm" if tem_mm else "px"
    cfg = {
        "area": st.column_config.TextColumn(t("results.col_area")),
        "detections": st.column_config.NumberColumn(t("results.col_detections"),
                                                    format="%d"),
        "pct": st.column_config.ProgressColumn(t("results.col_pct_detections"),
                                               min_value=0, max_value=100,
                                               format="%.1f%%"),
        "visits": st.column_config.NumberColumn(t("results.col_visits"), format="%d"),
        "residence": st.column_config.NumberColumn(t("results.col_residence_s"),
                                                   format="%.1f"),
        "dist": st.column_config.NumberColumn(t("results.col_dist_inside", unit=unid),
                                              format="%.1f" if tem_mm else "%.0f"),
        "area_mm2": st.column_config.NumberColumn(t("results.col_area_mm2"),
                                                  format="%.0f"),
    }
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={k: v for k, v in cfg.items() if k in df.columns})
    st.caption(t("results.zones_caption"))

    col1, col2 = st.columns(2)
    with col1:
        _mostrar(analytics.plot_mapa_zonas(centroides, fundo, zones))
    with col2:
        _mostrar(analytics.plot_permanencia_zonas(metr_z))


def _limiar_parada_txt(limiar_parada_px, pixels_per_mm=None, fps_video=None,
                       frame_skip=None):
    """Legenda do limiar de parada em uso (px, e mm / mm/s quando há escala).

    O único controle é o slider da barra lateral; aqui só mostramos o valor e
    onde alterá-lo (evita dois widgets disputando o mesmo valor).
    """
    equiv = ""
    if pixels_per_mm and pixels_per_mm > 0:
        mm = limiar_parada_px / pixels_per_mm
        if fps_video and fps_video > 0 and frame_skip:
            equiv = t("results.stop_threshold_equiv_speed", mm=mm,
                      mms=mm / (frame_skip / fps_video))
        else:
            equiv = t("results.stop_threshold_equiv", mm=mm)
    return t("results.stop_threshold_in_use", px=limiar_parada_px, equiv=equiv,
             where=f"{t('sidebar.settings')} → {t('sidebar.tracking_areas')}",
             label=t("sidebar.stop_threshold"))


# Gráficos opcionais: valor interno -> chave booleana em session_state
# (os rótulos traduzidos só aparecem via format_func).
_GRAF_OPCIONAIS = {"position": "graf_posicao", "speed": "graf_velocidade",
                   "hist": "graf_histograma"}


def _pills_graficos(opcoes, key):
    """Pills (multi) dos gráficos opcionais; mantém graf_* em session_state."""
    rotulos = {"position": t("results.opt_position"), "speed": t("results.opt_speed"),
               "hist": t("results.opt_speed_hist")}
    padrao = [o for o in opcoes if st.session_state.get(_GRAF_OPCIONAIS[o], True)]
    escolha = st.pills(t("results.optional_charts"), opcoes, selection_mode="multi",
                       default=padrao, format_func=lambda o: rotulos[o], key=key)
    escolha = list(escolha or [])
    for o in opcoes:
        st.session_state[_GRAF_OPCIONAIS[o]] = o in escolha
    return set(escolha)


def render_analytics(centroides, indices, fundo, fps_video, limiar_parada_px,
                     frames_processados=None, frame_skip=None, pixels_per_mm=None,
                     zones=None, area_ref_px2=None, key_prefix=""):
    """Cartões de métricas + grupos de gráficos a partir dos dados brutos.

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
        cobertura_txt = t("results.coverage_of_area", pct=pct)
    elif metr.get("cobertura_perc") is not None:
        cobertura_txt = t("results.coverage_of_frame", pct=metr['cobertura_perc'])
    else:
        cobertura_txt = None
    with st.expander(t("results.summary_metrics"), expanded=True):
        if tem_mm:
            st.caption(t("results.calibrated_scale", ppm=metr['pixels_per_mm']))
        a, b, c, d = st.columns(4)
        a.metric(t("results.detections"), metr["deteccoes"])
        if tem_mm:
            b.metric(t("results.total_distance"), f"{metr['distancia_mm']:.1f} mm")
            c.metric(t("results.net_displacement"), f"{metr['deslocamento_liquido_mm']:.1f} mm")
        else:
            b.metric(t("results.total_distance"), f"{metr.get('distancia_px', 0):.0f} px")
            c.metric(t("results.net_displacement"),
                     f"{metr.get('deslocamento_liquido_px', 0):.0f} px")
        d.metric(t("results.straightness"), f"{metr.get('retilineidade', 0):.2f}",
                 help=t("results.straightness_help"))
        e, f, g, h = st.columns(4)
        if tem_mm and tem_vel_mm:
            e.metric(t("results.mean_speed"), f"{metr['velocidade_media_mm_s']:.2f} mm/s")
            f.metric(t("results.max_speed"), f"{metr['velocidade_max_mm_s']:.2f} mm/s")
        else:
            e.metric(t("results.mean_speed"), f"{metr.get('velocidade_media', 0):.1f} {uv}")
            f.metric(t("results.max_speed"), f"{metr.get('velocidade_max', 0):.1f} {uv}")
        g.metric(t("results.time_moving"), f"{metr.get('perc_movimento', 0):.0f}%",
                 help=t("results.time_moving_help", thr=limiar_parada_px))
        if tem_mm:
            h.metric(t("results.explored_area"), f"{metr['area_explorada_mm2']:.1f} mm²",
                     delta=cobertura_txt, delta_color="off")
        else:
            h.metric(t("results.explored_area"), f"{metr.get('area_explorada_px2', 0):.0f} px²",
                     delta=cobertura_txt, delta_color="off")

        # Detalhamento de tempo (detectado/não detectado, caminhando/parado)
        if "tempo_detectado" in metr:
            st.markdown(t("results.time_breakdown"))
            t1, t2, t3, t4 = st.columns(4)
            t1.metric(t("results.time_detected"), f"{metr['tempo_detectado']:.1f} {ut}",
                      delta=t("results.pct_of_frames", pct=metr.get('perc_detectado', 0)),
                      delta_color="off")
            t2.metric(t("results.time_not_detected"), f"{metr['tempo_nao_detectado']:.1f} {ut}",
                      help=t("results.time_not_detected_help"))
            t3.metric(t("results.time_walking"), f"{metr.get('tempo_movimento', 0):.1f} {ut}")
            t4.metric(t("results.time_stopped"), f"{metr.get('tempo_parado', 0):.1f} {ut}")
        # Limiar de parada em uso: as métricas de tempo em movimento dependem dele.
        st.caption(_limiar_parada_txt(limiar_parada_px, pixels_per_mm, fps_video,
                                      frame_skip))

    if fundo is None:
        st.info(t("results.no_background"))
        return

    with st.expander(t("results.charts"), expanded=True):
        # Só o grupo escolhido é calculado (st.tabs desenharia todos).
        grupos = ["trajectory", "speed", "coverage"] + (["zones"] if zones else [])
        rotulos_grupo = {"trajectory": t("results.tab_trajectory"),
                         "speed": t("results.tab_speed_time"),
                         "coverage": t("results.tab_coverage"),
                         "zones": t("results.tab_zones")}
        grupo = st.segmented_control(
            t("results.chart_group"), grupos, default="trajectory", required=True,
            format_func=lambda g: rotulos_grupo[g], key=f"{key_prefix}graf_grupo",
            label_visibility="collapsed",
        )
        if grupo not in grupos:
            grupo = "trajectory"

        if grupo == "trajectory":
            extras = _pills_graficos(["position"], f"{key_prefix}graf_extra_traj")
            _mostrar(analytics.plot_trajetoria_tempo(centroides, indices, fundo,
                                                     fps_video, zones=zones),
                     t("results.trajectory_needs_2"))
            if "position" in extras:
                st.markdown(t("results.position_xy_time"))
                _mostrar(analytics.plot_posicao_tempo(centroides, indices, fps_video))

        elif grupo == "speed":
            extras = _pills_graficos(["speed", "hist"], f"{key_prefix}graf_extra_vel")
            if "speed" in extras:
                _mostrar(analytics.plot_velocidade_tempo(centroides, indices, fps_video))
            if "hist" in extras:
                col1, col2 = st.columns(2)
                with col1:
                    _mostrar(analytics.plot_distancia_acumulada(centroides, indices, fps_video))
                with col2:
                    _mostrar(analytics.plot_histograma_velocidade(centroides, indices, fps_video))
            else:
                _mostrar(analytics.plot_distancia_acumulada(centroides, indices, fps_video))

        elif grupo == "coverage":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(t("results.heatmap_title"))
                _mostrar(analytics.plot_heatmap(centroides, fundo, zones=zones))
            with col2:
                st.markdown(t("results.explored_area_title"))
                _mostrar(analytics.plot_area_explorada(centroides, fundo),
                         t("results.explored_area_needs_3"))

        elif grupo == "zones":
            render_zonas(centroides, indices, fundo, fps_video, zones,
                         frame_skip, pixels_per_mm)


def _tabela_placas(tids, rotulos, centroides, indices, track_ids, fundo, fps_video,
                   limiar_parada_px, frames_processados, frame_skip, pixels_per_mm,
                   zones, key_prefix):
    """Tabela-resumo (uma linha por placa) com as métricas de calcular_metricas.

    Valores crus + column_config (a ordenação funciona); uma unidade por coluna.
    As colunas extras ficam atrás do toggle "Mostrar todas as métricas".
    """
    linhas, metricas = [], []
    for tid in tids:
        c, idx = analytics.subset_track(centroides, indices, track_ids, tid)
        m = analytics.calcular_metricas(
            c, idx, fps_video, limiar_parada_px,
            frame_shape=fundo.shape if fundo is not None else None,
            frames_processados=frames_processados, frame_skip=frame_skip,
            pixels_per_mm=pixels_per_mm,
        )
        metricas.append(m)
        area_ref = (analytics._area_poligono_px2(zones[tid]["points"])
                    if zones and tid < len(zones) else None)
        cobertura = (100.0 * m["area_explorada_px2"] / area_ref
                     if area_ref and "area_explorada_px2" in m else m.get("cobertura_perc"))
        linhas.append((tid, m, cobertura))

    tem_mm = bool(pixels_per_mm and pixels_per_mm > 0)
    tem_vel_mm = tem_mm and any("velocidade_media_mm_s" in m for m in metricas)
    ud = "mm" if tem_mm else "px"
    ua = "mm²" if tem_mm else "px²"
    uv = "mm/s" if tem_vel_mm else next(
        (m["unidade_v"] for m in metricas if "unidade_v" in m), "px/s")
    ut = next((m["unidade_t"] for m in metricas if "unidade_t" in m), "s")
    suf = "_mm" if tem_mm else "_px"
    suf_v = "_mm_s" if tem_vel_mm else ""
    suf_a = "_mm2" if tem_mm else "_px2"

    dados = []
    for tid, m, cobertura in linhas:
        dados.append({
            "dish": rotulos[tid],
            "detections": m.get("deteccoes", 0),
            "distance": m.get("distancia" + suf),
            "speed_mean": m.get("velocidade_media" + suf_v),
            "pct_moving": m.get("perc_movimento"),
            "straightness": m.get("retilineidade"),
            "net": m.get("deslocamento_liquido" + suf),
            "speed_max": m.get("velocidade_max" + suf_v),
            "time_moving": m.get("tempo_movimento"),
            "time_stopped": m.get("tempo_parado"),
            "time_detected": m.get("tempo_detectado"),
            "pct_detected": m.get("perc_detectado"),
            "area": m.get("area_explorada" + suf_a),
            "coverage": cobertura,
        })
    df = pd.DataFrame(dados)
    nc = st.column_config.NumberColumn
    pc = st.column_config.ProgressColumn
    cfg = {
        "dish": st.column_config.TextColumn(t("results.col_dish")),
        "detections": nc(t("results.col_detections"), format="%d"),
        "distance": nc(t("results.col_distance", unit=ud),
                       format="%.1f" if tem_mm else "%.0f"),
        "speed_mean": nc(t("results.col_mean_speed", unit=uv), format="%.2f"),
        "pct_moving": pc(t("results.col_pct_moving"), min_value=0, max_value=100,
                         format="%.0f%%", help=t("results.time_moving_help",
                                                 thr=limiar_parada_px)),
        "straightness": nc(t("results.col_straightness"), format="%.2f",
                           help=t("results.straightness_help")),
        "net": nc(t("results.col_net_displacement", unit=ud),
                  format="%.1f" if tem_mm else "%.0f"),
        "speed_max": nc(t("results.col_max_speed", unit=uv), format="%.2f"),
        "time_moving": nc(t("results.col_time_walking", unit=ut), format="%.1f"),
        "time_stopped": nc(t("results.col_time_stopped", unit=ut), format="%.1f"),
        "time_detected": nc(t("results.col_time_detected", unit=ut), format="%.1f"),
        "pct_detected": pc(t("results.col_pct_detected"), min_value=0, max_value=100,
                           format="%.0f%%"),
        "area": nc(t("results.col_explored_area", unit=ua),
                   format="%.1f" if tem_mm else "%.0f"),
        "coverage": pc(t("results.col_coverage"), min_value=0, max_value=100,
                       format="%.1f%%"),
    }
    principais = ["dish", "detections", "distance", "speed_mean", "pct_moving",
                  "straightness"]
    todas = st.toggle(t("results.show_all_metrics"), key=f"{key_prefix}todas_metricas")
    ordem = [c for c in (list(df.columns) if todas else principais)
             if c in df.columns and not df[c].isna().all()]
    st.dataframe(df, hide_index=True, width="stretch", column_order=ordem,
                 column_config=cfg)


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
                         area_ref_px2=area_ref, key_prefix=key_prefix)
        return

    # Quando as áreas guiaram o rastreamento, rotula cada placa pelo nome da área.
    # (variável `tid`, não `t`: `t` é a função de tradução do i18n)
    def _rotulo(tid):
        if zones and tid < len(zones):
            return zones[tid].get("name") or t("results.bee_n", n=tid + 1)
        return t("results.bee_n", n=tid + 1)

    rotulos = {tid: _rotulo(tid) for tid in tids}

    with st.expander(t("results.overview_multi", n=len(tids)), expanded=True):
        if fundo is not None:
            _mostrar(analytics.plot_trajetorias_multi(
                centroides, indices, track_ids, fundo, fps_video,
                zones=zones, rotulos=rotulos))
        _tabela_placas(tids, rotulos, centroides, indices, track_ids, fundo,
                       fps_video, limiar_parada_px, frames_processados, frame_skip,
                       pixels_per_mm, zones, key_prefix)
        st.caption(_limiar_parada_txt(limiar_parada_px, pixels_per_mm, fps_video,
                                      frame_skip))

    # Detalhamento completo de uma placa escolhida (opções = ids estáveis,
    # o rótulo traduzido só aparece via format_func)
    tid = st.pills(t("results.show_details_for"), list(tids), default=tids[0],
                   required=True, format_func=lambda k: rotulos[k],
                   key=f"{key_prefix}sel_track_detalhe")
    if tid not in tids:
        tid = tids[0]
    c, idx = analytics.subset_track(centroides, indices, track_ids, tid)
    st.markdown(t("results.details_title", label=rotulos[tid]))
    # Recorta o fundo na caixa da área (a abelha ocupa o quadro, não um ponto
    # minúsculo no frame inteiro). Distância é invariante à translação, então as
    # métricas não mudam; zones=None evita redesenhar o contorno em coords do
    # frame inteiro (o recorte já É a área).
    fundo_t, c_t = (_recorte_area(fundo, c, zones[tid])
                    if zones and tid < len(zones) else (fundo, c))
    # Cobertura relativa à área delimitada (polígono da zona), não ao quadro.
    area_ref = (analytics._area_poligono_px2(zones[tid]["points"])
                if zones and tid < len(zones) else None)
    render_analytics(c_t, idx, fundo_t, fps_video, limiar_parada_px,
                     frames_processados=frames_processados, frame_skip=frame_skip,
                     pixels_per_mm=pixels_per_mm, zones=None, area_ref_px2=area_ref,
                     key_prefix=f"{key_prefix}det_")


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
    """CSV da trajetória. O cabeçalho é FIXO (placa,frame,x,y) em qualquer
    idioma da interface: é um formato para scripts (pandas/R) e para juntar
    exportações de usuários diferentes. Só o nome do arquivo é traduzido."""
    dados = {"frame": indices_frames, "x": centroides[:, 0], "y": centroides[:, 1]}
    if track_ids is not None and len(np.asarray(track_ids).reshape(-1)):
        dados = {"placa": np.asarray(track_ids).reshape(-1) + 1, **dados}
    return pd.DataFrame(dados).to_csv(index=False)


# --- Telas ---

def _entrar(user: dict):
    """Abre a sessão do usuário autenticado."""
    st.session_state["user_id"] = user["id"]
    st.session_state["username"] = user["username"]
    st.session_state["role"] = user.get("role", "user")


def tela_login():
    ui.inject_css()
    _, centro, _ = st.columns([1, 1.4, 1])
    with centro:
        i18n.language_selector()
        ui.header(t("core.app_name"), t("core.app_subtitle"))
        st.markdown(t("history.login_about"))
        st.caption(t("history.login_caption"))

        aba_entrar, aba_registrar = st.tabs([t("history.tab_login"), t("history.tab_register")])

        with aba_entrar:
            with st.form("form_login"):
                usuario = st.text_input(t("history.username"))
                senha = st.text_input(t("history.password"), type="password")
                if st.form_submit_button(t("history.login_button"), type="primary",
                                         width="stretch"):
                    user, erro = auth.login(usuario, senha)
                    if user:
                        _entrar(user)
                        st.rerun()
                    else:
                        st.error(erro)

        with aba_registrar:
            with st.form("form_registro"):
                usuario = st.text_input(t("history.username"), help=t("history.username_help"))
                senha = st.text_input(t("history.password"), type="password",
                                      help=t("history.password_help"))
                confirma = st.text_input(t("history.confirm_password"), type="password")
                if st.form_submit_button(t("history.register_button"), width="stretch"):
                    if senha != confirma:
                        st.error(t("history.passwords_mismatch"))
                    else:
                        ok, msg = auth.register(usuario, senha)
                        if not ok:
                            st.error(msg)
                        else:
                            # Conta criada: já entra com as mesmas credenciais.
                            user, erro = auth.login(usuario, senha)
                            if user:
                                _entrar(user)
                                ui.flash(t("history.register_welcome",
                                           user=user["username"]))
                                st.rerun()
                            else:
                                st.success(msg)
                                st.error(erro)


def _md_esc(texto):
    """Escapa caracteres de markdown em textos do usuário (ex.: nome do vídeo)."""
    s = str(texto)
    for ch in "\\`*_[]#<>|~":
        s = s.replace(ch, "\\" + ch)
    return s


def _fmt_data(created_at):
    """'2026-11-10 14:33:02' -> data/hora curta no formato do idioma."""
    try:
        return datetime.strptime(str(created_at)[:19], "%Y-%m-%d %H:%M:%S").strftime(
            t("history.date_strftime"))
    except (TypeError, ValueError):
        return str(created_at or "—")


def _duracao_analisada(frames_processados, frame_skip, fps_video):
    """Duração (s) do trecho analisado: amostras × intervalo entre amostras."""
    try:
        if fps_video and fps_video > 0 and frames_processados and frame_skip:
            return float(frames_processados) * float(frame_skip) / float(fps_video)
    except (TypeError, ValueError):
        pass
    return None


def _linha_fatos(fatos):
    """Linha compacta de fatos (rótulo pequeno + valor) num contêiner com borda."""
    with st.container(border=True):
        for col, (rotulo, valor) in zip(st.columns(len(fatos)), fatos):
            col.caption(rotulo)
            col.markdown(f"**{valor}**")


def _parametros_tecnicos(itens):
    """Expander 'Parâmetros técnicos' com os itens em duas colunas."""
    with st.expander(t("history.tech_params")):
        meio = (len(itens) + 1) // 2
        for col, parte in zip(st.columns(2), (itens[:meio], itens[meio:])):
            col.markdown("\n".join(f"- {rot}: **{val}**" for rot, val in parte))


def _nome_trajetoria(analysis_id):
    if analysis_id is None:
        return t("history.file_trajectory_noid")
    return t("history.file_trajectory", id=analysis_id)


def exibir_resultados(res: Resultado, frame_skip: int, limiar_parada_px: float,
                      pixels_per_mm: float | None = None, zones=None, conf=None):
    if len(res.centroides) == 0:
        st.warning(t("history.no_bees_detected"))
        # Dica com os rótulos exatos da barra lateral e os valores usados.
        st.info(t("analysis.no_bees_tips", settings=t("sidebar.settings"),
                  section=t("sidebar.detection"),
                  conf_label=t("sidebar.conf_threshold"),
                  conf=f"{conf:.2f}" if conf is not None else "—",
                  skip_label=t("sidebar.frame_skip"), skip=frame_skip,
                  zones_tool=t("analysis.tool_zones")))
        return

    analysis_id = (st.session_state.get("resultado_resumo") or {}).get("id")
    _parametros_tecnicos([
        (t("history.tp_processing_time"), _fmt_duracao(res.tempo_s)),
        (t("history.tp_processing_speed"),
         t("history.tp_fps_value", fps=res.fps_processamento)),
        (t("history.tp_frames"), t("history.tp_frames_value",
                                   done=res.frames_processados, total=res.total_frames)),
        (t("history.tp_frame_skip"), t("history.tp_frame_skip_value", skip=frame_skip)),
        (t("history.tp_conf"), f"{conf:.2f}" if conf is not None else "—"),
        (t("history.tp_video_fps"), f"{res.fps_video:.1f}" if res.fps_video else "—"),
    ])
    st.download_button(
        t("history.download_trajectory_csv"),
        data=csv_da_analise(res.centroides, res.indices_frames, res.track_ids),
        file_name=_nome_trajetoria(analysis_id), mime="text/csv", on_click="ignore",
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

    # Cabeçalho: fatos essenciais numa linha; detalhes técnicos no expander.
    st.markdown(t("history.analysis_title", id=analise['id']))
    n_placas = (len(zonas_salvas) if zonas_salvas
                else analise.get("num_targets") or 1)
    duracao = _duracao_analisada(analise.get("frames_processed"),
                                 analise.get("frame_skip"), analise.get("fps_video"))
    _linha_fatos([
        (t("history.fact_video"), _md_esc(analise["video_name"])),
        (t("history.fact_date"), _fmt_data(analise["created_at"])),
        (t("history.fact_dishes"), t("history.fact_dishes_value", n=n_placas)
         if zonas_salvas else str(n_placas)),
        (t("history.fact_scale"), f"{escala_salva:.2f} px/mm" if escala_salva
         else t("history.fact_no_scale")),
        (t("history.fact_duration"), _fmt_duracao(duracao) if duracao is not None
         else "—"),
    ])

    def _num(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "—"

    _parametros_tecnicos([
        (t("history.tp_backend"), rotulo_backend(analise.get("backend"))),
        (t("history.tp_input_size"), _num(analise.get("input_size"), "{} px")),
        (t("history.tp_conf"), _num(analise.get("conf_threshold"), "{:.2f}")),
        (t("history.tp_nms"), _num(analise.get("nms_threshold"), "{:.2f}")),
        (t("history.tp_frame_skip"),
         t("history.tp_frame_skip_value", skip=analise.get("frame_skip") or "—")),
        (t("history.tp_frames"),
         t("history.tp_frames_value", done=analise.get("frames_processed") or 0,
           total=analise.get("total_frames") or 0)),
        (t("history.tp_video_fps"), _num(analise.get("fps_video"), "{:.1f}")),
        (t("history.tp_processing_time"),
         _fmt_duracao(analise["processing_time_s"])
         if analise.get("processing_time_s") is not None else "—"),
        (t("history.tp_processing_speed"),
         t("history.tp_fps_value", fps=analise["fps_processing"])
         if analise.get("fps_processing") else "—"),
    ])

    if centroides is not None and len(centroides) > 0:
        st.download_button(
            t("history.download_trajectory_csv"),
            data=csv_da_analise(centroides, indices, track_ids),
            file_name=_nome_trajetoria(analise["id"]),
            mime="text/csv", on_click="ignore",
        )
        render_resultados(centroides, indices, track_ids, fundo,
                          analise["fps_video"], limiar_parada_px,
                          frames_processados=analise["frames_processed"],
                          frame_skip=analise["frame_skip"],
                          pixels_per_mm=escala_salva, zones=zonas_salvas,
                          key_prefix=f"hist{analise['id']}_")
    else:
        st.info(t("history.no_detections_saved"))

    st.markdown("---")
    if st.button(t("history.delete_analysis"), key=f"excluir_{analise['id']}"):
        def _excluir(aid=analise["id"]):
            database.delete_analysis(aid, user_id)
            log_event("analysis_deleted", f"Análise #{aid} excluída",
                      user_id=user_id, analysis_id=aid)
            if st.session_state.get("_keep_hist_analise") == aid:
                st.session_state.pop("_keep_hist_analise", None)
            if (st.session_state.get("conf_ctx") or {}).get("analysis_id") == aid:
                st.session_state.pop("conf_ctx", None)  # Conferência desta análise
        ui.confirm_delete(
            t("history.delete_dialog_title", id=analise["id"]),
            t("history.delete_dialog_body", id=analise["id"],
              video=ui.esc(analise["video_name"]), created=analise["created_at"]),
            _excluir, t("history.delete_done", id=analise["id"]),
            key="dlg_excluir_analise")


def _tabela_historico(analises) -> pd.DataFrame:
    """Resumo das análises com valores crus (datas/números), colunas internas."""
    linhas = []
    for a in analises:
        escala = a.get("pixels_per_mm")
        dist_px = a.get("distance_px")
        linhas.append({
            "id": a["id"],
            "created_at": pd.to_datetime(a.get("created_at"), errors="coerce"),
            "video_name": a.get("video_name"),
            "dishes": a.get("num_targets") or 1,
            "detections": a.get("detections"),
            "distance_mm": (dist_px / escala) if (dist_px is not None and escala
                                                  and escala > 0) else None,
            "distance_px": dist_px,
            "duration_s": _duracao_analisada(a.get("frames_processed"),
                                             a.get("frame_skip"), a.get("fps_video")),
            "scale": escala if escala and escala > 0 else None,
            "frame_skip": a.get("frame_skip"),
            "conf": a.get("conf_threshold"),
            "backend": rotulo_backend(a.get("backend")),
            "processing_time_s": a.get("processing_time_s"),
        })
    return pd.DataFrame(linhas)


def _config_historico():
    nc = st.column_config.NumberColumn
    return {
        "id": nc(t("history.col_id"), format="#%d", width="small"),
        "created_at": st.column_config.DatetimeColumn(
            t("history.col_date"), format=t("history.date_column_format")),
        "video_name": st.column_config.TextColumn(t("history.col_video")),
        "dishes": nc(t("history.col_dishes"), format="%d"),
        "detections": nc(t("history.col_detections"), format="%d"),
        "distance_mm": nc(t("history.col_distance", unit="mm"), format="%.1f"),
        "distance_px": nc(t("history.col_distance", unit="px"), format="%.0f"),
        "duration_s": nc(t("history.col_duration_s"), format="%.0f"),
        "scale": nc(t("history.col_scale"), format="%.2f"),
        "frame_skip": nc(t("history.col_frame_skip"), format="%d"),
        "conf": nc(t("history.col_conf"), format="%.2f"),
        "backend": st.column_config.TextColumn(t("history.col_backend")),
        "processing_time_s": nc(t("history.col_time_s"), format="%.1f"),
    }


def _csv_historico(df: pd.DataFrame) -> bytes:
    """CSV do resumo: valores crus, cabeçalhos traduzidos com unidade."""
    rotulos = {k: v["label"] if isinstance(v, dict) else k
               for k, v in _config_historico().items()}
    out = df.copy()
    out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.rename(columns=rotulos).to_csv(index=False).encode("utf-8")


def _exportar_todas(user_id, username, analises, df, limiar_parada_px):
    """Popover com as exportações de TODAS as análises (CSV / Excel / PDF).

    Os relatórios gerados ficam em session_state junto de uma "impressão
    digital" (análises + limiar + idioma); se o conjunto mudar, são descartados.
    """
    hoje = datetime.now().strftime("%Y-%m-%d")
    impressao = (tuple(a["id"] for a in analises), float(limiar_parada_px),
                 i18n.get_lang())
    for chave in ("relatorio_xlsx", "relatorio_pdf"):
        guardado = st.session_state.get(chave)
        if not (isinstance(guardado, tuple) and guardado[0] == impressao):
            st.session_state.pop(chave, None)

    with st.popover(t("history.export_all"), width="stretch"):
        st.caption(t("history.export_all_caption", n=len(analises)))
        col_csv, col_xlsx, col_pdf = st.columns(3)
        with col_csv:
            st.download_button(
                t("history.summary_csv"), data=_csv_historico(df),
                file_name=t("history.file_summary_csv", date=hoje), mime="text/csv",
                width="stretch", on_click="ignore", key="exp_csv",
            )

        formatos = [
            (col_xlsx, "relatorio_xlsx", "xlsx", report.build_excel,
             t("history.generate_excel"), t("history.download_excel"),
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "report_excel_generated", "history.excel_failed"),
            (col_pdf, "relatorio_pdf", "pdf", report.build_pdf,
             t("history.generate_pdf"), t("history.download_pdf"),
             "application/pdf", "report_generated", "history.report_failed"),
        ]
        for col, chave, ext, construir, rot_gerar, rot_baixar, mime, evento, erro_key \
                in formatos:
            with col:
                slot = st.empty()
                if chave not in st.session_state:
                    if not slot.button(rot_gerar, width="stretch", key=f"gerar_{ext}"):
                        continue
                    with st.spinner(t("history.building_report")):
                        try:
                            dados = construir(database.list_analyses_full(user_id),
                                              username, limiar_parada_px)
                            st.session_state[chave] = (impressao, dados)
                            log_event(evento,
                                      f"Relatório {ext.upper()} gerado "
                                      f"({len(analises)} análises)", user_id=user_id)
                        except Exception as e:
                            slot.error(t(erro_key, error=e))
                            log_event("report_failed", f"Erro no relatório: {e}",
                                      user_id=user_id, level=logging.ERROR)
                            continue
                slot.download_button(
                    rot_baixar, data=st.session_state[chave][1],
                    file_name=t("history.file_report", date=hoje) + f".{ext}",
                    mime=mime, width="stretch", on_click="ignore", type="primary",
                    key=f"baixar_{ext}",
                )


def render_historico(user_id: int, username: str, limiar_parada_px: float):
    analises = database.list_analyses(user_id)
    if not analises:
        st.info(t("history.empty"))
        if st.button(t("analysis.go_new"), type="primary", key="hist_ir_nova"):
            go_to("new")
    else:
        df_todas = _tabela_historico(analises)
        c_filtro, c_exportar = st.columns([3, 1], vertical_alignment="bottom")
        filtro = c_filtro.text_input(
            t("history.filter_label"), key="hist_filtro",
            placeholder=t("history.filter_placeholder"), icon=":material/search:",
        )
        with c_exportar:
            _exportar_todas(user_id, username, analises, df_todas, limiar_parada_px)

        termo = (filtro or "").strip().lower()
        df = (df_todas[df_todas["video_name"].fillna("").str.lower().str.contains(
            termo, regex=False)] if termo else df_todas).reset_index(drop=True)
        ids = [int(i) for i in df["id"]]
        st.caption(t("history.count_caption", n=len(analises)) if not termo else
                   t("history.count_filtered", n=len(ids), total=len(analises)))
        if not ids:
            st.info(t("history.filter_no_match", term=filtro.strip()))
        else:
            # A tabela é o seletor. A chave muda só quando o conjunto exibido muda
            # (filtro, análise nova/excluída): aí a seleção recomeça a partir da
            # análise aberta (selection_default), em vez de apontar p/ outra linha.
            assinatura = zlib.crc32(",".join(map(str, ids)).encode())
            chave_tbl = f"hist_tbl_{assinatura}"
            aberta = st.session_state.get("_keep_hist_analise")
            linha_padrao = ids.index(aberta) if aberta in ids else 0
            visiveis = ["id", "created_at", "video_name", "dishes", "detections"]
            if df["distance_mm"].notna().any():
                visiveis.append("distance_mm")
            visiveis += ["distance_px", "duration_s"]
            evento = st.dataframe(
                df, hide_index=True, width="stretch", key=chave_tbl,
                on_select="rerun", selection_mode="single-row",
                selection_default={"selection": {"rows": [linha_padrao]}},
                column_order=visiveis, column_config=_config_historico(),
                height=min(38 + 35 * len(ids), 300),
            )
            try:
                linhas = list(evento["selection"]["rows"])
            except (KeyError, TypeError):
                linhas = []
            if linhas and 0 <= linhas[0] < len(ids):
                analise_id = ids[linhas[0]]
                st.session_state["_keep_hist_analise"] = analise_id
            else:
                # Nada selecionado: abre a mais recente (lista já vem ordenada).
                analise_id = ids[0]
                st.caption(t("history.select_hint"))
            analise = database.get_analysis(analise_id, user_id)
            if analise:
                st.markdown("---")
                exibir_analise_salva(analise, user_id, limiar_parada_px)

    with st.expander(t("history.activity_log")):
        eventos = database.list_logs(user_id=user_id, limit=100)
        if eventos:
            _tabela_atividade(eventos)
        else:
            st.caption(t("history.no_events"))


def _tabela_atividade(eventos):
    """Log de atividade da conta com rótulos traduzidos.

    O evento (código estável, ex.: "analysis_completed") vira uma descrição
    no idioma atual (log.<evento>; sem tradução, mostra o próprio código). O
    texto livre do registro é gravado em português, então só aparece em pt."""
    catalogo = i18n.catalog(i18n.get_lang())
    niveis = {"INFO": t("history.log_level_info"),
              "WARNING": t("history.log_level_warning"),
              "ERROR": t("history.log_level_error")}
    df = pd.DataFrame(eventos)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["level"] = df["level"].map(lambda n: niveis.get(str(n), n))
    df["event"] = df["event"].map(
        lambda e: t(f"log.{e}") if f"log.{e}" in catalogo else e)
    colunas = ["ts", "level", "event"]
    if i18n.get_lang() == "pt":
        colunas.append("message")
    st.dataframe(df, hide_index=True, width="stretch", column_order=colunas,
                 column_config={
                     "ts": st.column_config.DatetimeColumn(
                         t("history.col_date"), format=t("history.date_column_format")),
                     "level": st.column_config.TextColumn(t("history.col_log_level")),
                     "event": st.column_config.TextColumn(t("history.col_log_event")),
                     "message": st.column_config.TextColumn(
                         t("history.col_log_details")),
                 })


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


def _fmt_duracao(s):
    """Segundos -> duração curta legível ('8.3 s', '2 min 14 s', '1 h 05 min')."""
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    s = int(round(s))
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def _fps_video(video_path):
    """FPS do vídeo (cacheado por sessão) — para estimar os frames amostrados."""
    cache = st.session_state.get("_fps_cache")
    if cache and cache[0] == video_path:
        return cache[1]
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    st.session_state["_fps_cache"] = (video_path, fps)
    return fps


CANVAS_MAX_W = 600   # largura máxima dos canvases de desenho (cabe em laptops)


def ferramenta_calibracao(video_path):
    """Permite desenhar uma linha sobre objeto de medida conhecida -> px/mm."""
    try:
        _patch_canvas_compat()
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception:
        st.info(t("calib.canvas_unavailable"))
        st.session_state["pixels_per_mm"] = st.number_input(
            t("calib.scale_input"), min_value=0.0,
            value=float(st.session_state.get("pixels_per_mm", 0.0)),
            step=0.1, format="%.2f",
        )
        return

    frame = _primeiro_frame(video_path)
    if frame is None:
        st.warning(t("calib.first_frame_error"))
        return

    altura, largura = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 1º o comprimento real (o que o usuário precisa saber antes de desenhar).
    col_mm, _ = st.columns([1, 2])
    with col_mm:
        comprimento_mm = st.number_input(
            t("calib.real_length"), min_value=0.01, value=100.0, step=0.1,
            key="comprimento_mm", help=t("calib.real_length_help"))
    st.markdown(t("calib.instructions", compute=t("calib.compute_scale")))

    canvas_w = max(1, min(largura, CANVAS_MAX_W))
    canvas_h = int(altura * canvas_w / largura)
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)", stroke_width=3, stroke_color="#FF0000",
        background_image=pil, update_streamlit=True,
        height=canvas_h, width=canvas_w, drawing_mode="line", key="canvas_calib",
    )
    escala = st.session_state.get("pixels_per_mm", 0.0)
    col_calc, col_limpar, _ = st.columns([1, 1, 2])
    with col_calc:
        calcular = st.button(t("calib.compute_scale"), type="primary", width="stretch")
    with col_limpar:
        limpar = st.button(t("calib.clear_scale"), disabled=not escala > 0,
                           width="stretch")
    if calcular:
        objetos = (canvas.json_data or {}).get("objects") if canvas else None
        # O canvas ≥0.12 (Fabric.js 6) devolve o tipo com o nome da classe
        # ("Line"); o 0.9 (Fabric 4) devolvia "line".
        linhas = [o for o in (objetos or [])
                  if str(o.get("type", "")).lower() == "line"]
        if not linhas:
            st.error(t("calib.draw_line_first"))
        else:
            o = linhas[-1]
            # O comprimento (x2-x1, y2-y1) independe de left/top: o
            # deslocamento entre extremos é invariante ao centro do objeto.
            # scaleX/scaleY: a linha pode ter sido esticada no modo de edição.
            comp_canvas = float(np.hypot(
                (o["x2"] - o["x1"]) * float(o.get("scaleX") or 1),
                (o["y2"] - o["y1"]) * float(o.get("scaleY") or 1)))
            comp_imagem = comp_canvas * (largura / canvas_w)  # canvas -> resolução real
            if comp_imagem > 0:
                escala = comp_imagem / comprimento_mm
                st.session_state["pixels_per_mm"] = escala
                st.success(t("calib.scale_result", escala=escala,
                             px=comp_imagem, mm=comprimento_mm))
            else:
                st.error(t("calib.zero_length"))
    if limpar:
        st.session_state["pixels_per_mm"] = 0.0
        st.rerun()

    escala = st.session_state.get("pixels_per_mm", 0.0)
    if escala > 0:
        st.caption(t("calib.current_scale", escala=escala))
    else:
        st.caption(t("calib.no_scale"))


_ORIGEM = {"left": 0.0, "top": 0.0, "center": 0.5, "right": 1.0, "bottom": 1.0}
_COR_ZONA = {"stroke": "#FFA500", "fill": "rgba(255,165,0,0.25)", "strokeWidth": 2}


def _elipse_do_objeto(obj):
    """(cx, cy, rx, ry, ângulo°) em coords do CANVAS de um círculo/elipse do
    fabric.js, ou None se o objeto não for um.

    O CircleTool do streamlit-drawable-canvas usa o ponto do clique como ORIGEM
    (originX="left", originY="center" → ponto na circunferência) e GIRA o
    círculo por `angle` na direção do arraste; depois de editado ele também
    pode ter scaleX ≠ scaleY (elipse). Logo o centro é a origem mais a meia
    caixa (com o traço) girada por `angle` — a fórmula geral do fabric.
    """
    tipo = str(obj.get("type", "")).lower()
    sx = float(obj.get("scaleX") or 1.0)
    sy = float(obj.get("scaleY") or 1.0)
    if tipo == "circle":
        raio = float(obj.get("radius") or 0.0)
        rx0 = ry0 = raio
    elif tipo == "ellipse":
        rx0, ry0 = float(obj.get("rx") or 0.0), float(obj.get("ry") or 0.0)
    else:
        return None
    sw = float(obj.get("strokeWidth") or 0.0)
    ang_graus = float(obj.get("angle") or 0.0)
    ang = np.radians(ang_graus)
    ca, sa = np.cos(ang), np.sin(ang)
    dx = (0.5 - _ORIGEM.get(obj.get("originX", "left"), 0.0)) * (2 * rx0 + sw) * sx
    dy = (0.5 - _ORIGEM.get(obj.get("originY", "top"), 0.0)) * (2 * ry0 + sw) * sy
    left, top = float(obj.get("left") or 0.0), float(obj.get("top") or 0.0)
    return (left + dx * ca - dy * sa, top + dx * sa + dy * ca,
            rx0 * sx, ry0 * sy, ang_graus)


def _forma_do_objeto(obj, escala):
    """Geometria (px da IMAGEM) de um círculo/elipse do canvas, guardada na
    área para recarregá-la no canvas como forma editável; None p/ polígonos
    (esses se recriam a partir dos próprios vértices)."""
    elipse = _elipse_do_objeto(obj)
    if elipse is None:
        return None
    cx, cy, rx, ry, ang = elipse
    return {"tipo": "elipse", "cx": float(cx * escala), "cy": float(cy * escala),
            "rx": float(rx * escala), "ry": float(ry * escala), "angulo": float(ang)}


def _poligono_do_objeto(obj, escala, n_circulo=40):
    """Extrai os vértices (em px da imagem) de um objeto do canvas (fabric.js).

    Suporta polígonos, círculos/elipses e objetos 'path'. Um círculo vira um
    polígono de aproximação (`n_circulo` lados) — assim toda a lógica de zona
    (ponto-em-polígono, área, desenho) funciona sem casos especiais.
    Coordenadas do canvas são multiplicadas por `escala` (canvas -> imagem).
    """
    pts = []
    tipo = str(obj.get("type", "")).lower()
    elipse = _elipse_do_objeto(obj)
    if elipse is not None:
        cx, cy, rx, ry, ang_graus = elipse
        ang = np.radians(ang_graus)
        ca, sa = np.cos(ang), np.sin(ang)
        for k in range(n_circulo):
            th = 2.0 * np.pi * k / n_circulo
            ex, ey = rx * np.cos(th), ry * np.sin(th)  # ponto na elipse (eixos locais)
            pts.append([(cx + ex * ca - ey * sa) * escala,
                        (cy + ex * sa + ey * ca) * escala])
    elif tipo == "polygon" and obj.get("points"):
        left, top = obj.get("left", 0.0), obj.get("top", 0.0)
        sx = float(obj.get("scaleX") or 1.0)
        sy = float(obj.get("scaleY") or 1.0)
        # O fabric põe left/top na borda da caixa INCLUINDO meio traço
        # (left = min(x) - strokeWidth/2): descontamos para os vértices ficarem
        # exatos e a área não "andar" a cada salvar/recarregar.
        sw = float(obj.get("strokeWidth") or 0.0)
        xs = [p["x"] for p in obj["points"]]
        ys = [p["y"] for p in obj["points"]]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
        # Centro do objeto = origem + (meia caixa, com traço) girada por angle;
        # cada vértice = centro + (ponto - centro dos pontos) escalado e girado.
        ang = np.radians(float(obj.get("angle") or 0.0))
        ca, sa = np.cos(ang), np.sin(ang)
        dx = (0.5 - _ORIGEM.get(obj.get("originX", "left"), 0.0)) * (maxx - minx + sw) * sx
        dy = (0.5 - _ORIGEM.get(obj.get("originY", "top"), 0.0)) * (maxy - miny + sw) * sy
        cx, cy = left + dx * ca - dy * sa, top + dx * sa + dy * ca
        mx, my = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        for p in obj["points"]:
            lx, ly = (p["x"] - mx) * sx, (p["y"] - my) * sy
            pts.append([(cx + lx * ca - ly * sa) * escala,
                        (cy + lx * sa + ly * ca) * escala])
    elif obj.get("path"):
        for seg in obj["path"]:
            if len(seg) >= 3 and isinstance(seg[1], (int, float)):
                pts.append([seg[1] * escala, seg[2] * escala])
    return pts


def _circulo_para_poligono(cx, cy, r, n=40):
    """Converte um círculo (px da imagem) em polígono de `n` lados."""
    return [[float(cx + r * np.cos(2 * np.pi * k / n)),
             float(cy + r * np.sin(2 * np.pi * k / n))] for k in range(n)]


def _forma_circulo(cx, cy, r):
    """Geometria de elipse (px da imagem) de um círculo (cx, cy, r)."""
    return {"tipo": "elipse", "cx": float(cx), "cy": float(cy),
            "rx": float(r), "ry": float(r), "angulo": 0.0}


def _zonas_para_canvas(zonas, escala):
    """initial_drawing (fabric.js) com as áreas JÁ definidas, como formas
    editáveis: assim, ao voltar à ferramenta, o usuário vê e corrige (move,
    redimensiona, gira, exclui) o que salvou, em vez de um canvas vazio.

    Áreas com geometria de elipse ("forma") voltam como Circle com
    scaleX/scaleY (= elipse) e origem no centro — `_elipse_do_objeto` lê de
    volta sem ambiguidade; as demais voltam como Polygon dos vértices.
    Coordenadas da imagem são divididas por `escala` (imagem -> canvas). Nome
    de classe ("Circle", "Polygon"): é como o Fabric 6 do canvas as registra.
    """
    objetos = []
    for z in zonas or []:
        f = z.get("forma") or {}
        if f.get("tipo") == "elipse" and float(f.get("rx") or 0) > 0:
            rx, ry = float(f["rx"]), float(f.get("ry") or f["rx"])
            objetos.append({
                "type": "Circle", "left": f["cx"] / escala, "top": f["cy"] / escala,
                "radius": rx / escala, "scaleX": 1, "scaleY": ry / rx,
                "angle": float(f.get("angulo") or 0.0),
                "originX": "center", "originY": "center", **_COR_ZONA})
            continue
        pts = z.get("points") or []
        if len(pts) < 3:
            continue
        pontos = [{"x": float(x) / escala, "y": float(y) / escala} for x, y in pts]
        meio = _COR_ZONA["strokeWidth"] / 2.0
        objetos.append({
            "type": "Polygon", "points": pontos,
            "left": min(p["x"] for p in pontos) - meio,
            "top": min(p["y"] for p in pontos) - meio,
            "originX": "left", "originY": "top", "angle": 0,
            "scaleX": 1, "scaleY": 1, **_COR_ZONA})
    return {"version": "6.0.0", "objects": objetos}


def _zonas_do_canvas(objetos, escala, prefixo):
    """Áreas (px da imagem) a partir dos objetos do canvas, com a geometria
    de elipse quando a forma é um círculo/elipse."""
    zonas = []
    for o in objetos or []:
        pts = _poligono_do_objeto(o, escala)
        if len(pts) < 3:
            continue
        z = {"name": f"{prefixo} {len(zonas) + 1}", "points": pts}
        forma = _forma_do_objeto(o, escala)
        if forma:
            z["forma"] = forma
        zonas.append(z)
    return zonas


def _mesmas_zonas(a, b, tol=1.5):
    """True se as duas listas de áreas têm a mesma geometria (±`tol` px)."""
    if len(a or []) != len(b or []):
        return False
    for za, zb in zip(a or [], b or []):
        pa = np.asarray(za.get("points") or [], float).reshape(-1, 2)
        pb = np.asarray(zb.get("points") or [], float).reshape(-1, 2)
        if pa.shape != pb.shape or not np.allclose(pa, pb, atol=tol):
            return False
    return True


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

    def descartar(self):
        """Aborta a gravação (processamento interrompido) e apaga o MP4 parcial."""
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
        try:
            os.remove(self.path)
        except OSError:
            pass


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


def _modo_ajuste_canvas():
    """drawing_mode para mover/excluir formas, conforme a versão instalada do
    streamlit-drawable-canvas: "transform" (≤0.9), "edit" (0.10–0.11) ou None
    (≥0.12: a edição virou um botão da barra do canvas, ativo em qualquer modo,
    e passar "transform"/"edit" levanta ValueError)."""
    try:
        import streamlit_drawable_canvas as _sdc
    except Exception:
        return "transform"
    validos = getattr(_sdc, "_VALID_DRAWING_MODES", None)
    if not validos:
        return "transform"
    return next((m for m in ("transform", "edit") if m in validos), None)


def _chave_nome_zona(i):
    """Key do campo de nome da área `i` na geração atual das áreas."""
    return f"zona_nome_{st.session_state.get('_zonas_geracao', 0)}_{i}"


def _sincronizar_nomes_zonas(zonas=None):
    """Faz os campos de nome das áreas recomeçarem dos nomes das áreas recriadas.

    Um text_input com key ignora `value=` depois de criado: sem isto, ao
    recriar as áreas (detectar, salvar, excluir, limpar) os campos devolveriam
    os nomes antigos. Em vez de reescrever o estado dos widgets (o que falha
    quando o campo não está na tela), troca a "geração" das keys: os campos
    novos nascem com value = nome da área."""
    st.session_state["_zonas_geracao"] = st.session_state.get("_zonas_geracao", 0) + 1


def _excluir_zona(indice):
    """Callback do 🗑️ de uma área: remove-a mantendo os nomes das demais.

    Roda como on_click (antes do script): lê os nomes digitados até agora."""
    zonas = list(st.session_state.get("zones") or [])
    for i, z in enumerate(zonas):
        nome = st.session_state.get(_chave_nome_zona(i))
        if nome:
            z["name"] = nome
    if 0 <= indice < len(zonas):
        removida = zonas.pop(indice)
        ui.flash(t("analysis.zone_deleted", nome=removida.get("name") or indice + 1))
    st.session_state["zones"] = zonas
    _sincronizar_nomes_zonas(zonas)


def ferramenta_zonas(video_path):
    """Permite desenhar polígonos sobre o frame para definir áreas de monitoramento."""
    try:
        _patch_canvas_compat()
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception:
        st.info(t("analysis.canvas_unavailable_zones"))
        return

    frame = _primeiro_frame(video_path)
    if frame is None:
        st.warning(t("analysis.first_frame_unreadable_zones"))
        return

    altura, largura = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    zonas = st.session_state.get("zones") or []
    rotulo_salvar = t("analysis.save_drawn_zones")

    # --- Caminho principal: detecção automática das placas (Hough) ---
    st.markdown(t("analysis.hough_title"))
    st.caption(t("analysis.hough_caption"))
    col_n, col_btn = st.columns([1, 1], vertical_alignment="bottom")
    with col_n:
        n_placas_auto = st.number_input(
            t("analysis.hough_n"), min_value=1, max_value=30,
            value=max(1, min(30, len(zonas) or 1)), key="hough_n",
            help=t("analysis.hough_n_help"),
        )
    with col_btn:
        detectar = st.button(t("analysis.hough_detect"), type="primary",
                             width="stretch")
    with st.expander(t("analysis.hough_params")):
        param2 = st.slider(
            t("analysis.hough_sensitivity"), 10, 100, 30, key="hough_p2",
            help=t("analysis.hough_sensitivity_help"),
        )
        raio_min = st.slider(t("analysis.hough_rmin"), 1, 30, 5, key="hough_rmin",
                             help=t("analysis.hough_size_help")) / 100.0
        raio_max = st.slider(t("analysis.hough_rmax"), 5, 60, 25, key="hough_rmax",
                             help=t("analysis.hough_size_help")) / 100.0
    if detectar:
        circulos = _detectar_placas_hough(
            frame, int(n_placas_auto), param2, raio_min, raio_max)
        if not circulos:
            st.warning(t("analysis.hough_none", fine=t("analysis.hough_params"),
                         rigor=t("analysis.hough_sensitivity")))
        else:
            # Já salva as placas como áreas e as injeta no canvas para ajuste:
            # se o usuário corrigir alguma, clica em Salvar de novo.
            prefixo = (st.session_state.get("zonas_prefixo")
                       or t("analysis.zone_name_default"))
            st.session_state["zones"] = [
                {"name": f"{prefixo} {i + 1}", "points": _circulo_para_poligono(cx, cy, r),
                 "forma": _forma_circulo(cx, cy, r)}
                for i, (cx, cy, r) in enumerate(circulos)
            ]
            _sincronizar_nomes_zonas(st.session_state["zones"])
            st.session_state["zona_forcar_ajuste"] = True  # já entra no modo ajustar
            ui.flash(t("analysis.hough_found", n=len(circulos), save=rotulo_salvar))
            st.rerun()

    # --- Desenho / ajuste manual ---
    st.markdown(t("analysis.manual_title"))
    # Após "Detectar placas", já entra no modo de ajuste (definido antes do widget,
    # pois não se pode alterar o session_state de um widget já instanciado).
    # "transform" é o valor interno estável da opção Ajustar; o modo real do
    # canvas vem de _modo_ajuste_canvas(). Sem modo de ajuste (canvas ≥0.12),
    # a opção some e a edição fica no botão da barra do próprio canvas.
    modo_ajuste = _modo_ajuste_canvas()
    _formatos = {
        "polygon": t("analysis.zone_shape_polygon"),
        "circle": t("analysis.zone_shape_circle"),
    }
    if modo_ajuste:
        _formatos["transform"] = t("analysis.zone_shape_adjust")
    if st.session_state.get("zona_formato") not in (None, *_formatos):
        st.session_state.pop("zona_formato", None)  # valor antigo (rótulo)
    if st.session_state.pop("zona_forcar_ajuste", False):
        st.session_state["zona_formato"] = "transform" if modo_ajuste else "circle"
    # O Streamlit apaga o estado do rádio quando a ferramenta sai da tela;
    # restauramos a última escolha para não voltar sempre para "Polígono".
    if "zona_formato" not in st.session_state:
        _ult = st.session_state.get("_keep_zona_formato")
        if _ult in _formatos:
            st.session_state["zona_formato"] = _ult

    formato = st.radio(
        t("analysis.zone_shape"), list(_formatos), format_func=_formatos.get,
        horizontal=True, key="zona_formato",
    )
    if formato not in _formatos:
        formato = "circle"
    st.session_state["_keep_zona_formato"] = formato
    modo = modo_ajuste if formato == "transform" else formato
    if formato == "polygon":
        st.markdown(t("analysis.help_polygon"))
    elif formato == "circle":
        st.markdown(t("analysis.help_circle"))
    else:
        st.markdown(t("analysis.help_adjust"))
    if not modo_ajuste:
        # A barra do canvas só aparece com o mouse sobre a imagem: explicar
        # onde fica o lápis é o que torna a edição descobrível.
        st.info(t("analysis.help_toolbar_edit"), icon="✏️")

    canvas_w = max(1, min(largura, CANVAS_MAX_W))
    canvas_h = int(altura * canvas_w / largura)
    escala = largura / canvas_w

    # As áreas já definidas (desenhadas, salvas ou detectadas por Hough) voltam
    # ao canvas como formas editáveis — inclusive depois de trocar de
    # ferramenta ou de tela, quando o canvas é recriado vazio.
    init_draw = _zonas_para_canvas(zonas, escala)

    canvas = st_canvas(
        fill_color="rgba(255,165,0,0.25)", stroke_width=2, stroke_color="#FFA500",
        background_image=pil, update_streamlit=True, initial_drawing=init_draw,
        height=canvas_h, width=canvas_w, drawing_mode=modo, key="canvas_zonas",
    )
    col_pref, col_salvar, col_limpar = st.columns([2, 2, 1], vertical_alignment="bottom")
    with col_pref:
        prefixo = st.text_input(t("analysis.zone_name_prefix"),
                                value=t("analysis.zone_name_default"), key="zonas_prefixo")
    with col_salvar:
        salvar = st.button(rotulo_salvar, width="stretch")
    with col_limpar:
        limpar = st.button(t("analysis.clear_zones"), disabled=not zonas,
                           width="stretch")
    objetos = (canvas.json_data or {}).get("objects") if canvas else None
    no_canvas = _zonas_do_canvas(objetos, escala, prefixo)
    if salvar:
        if not no_canvas:
            st.error(t("analysis.draw_zone_first"))
        else:
            # Mesma quantidade de áreas → é um ajuste: preserva os nomes dados.
            if len(no_canvas) == len(zonas):
                for z_novo, z_ant in zip(no_canvas, zonas):
                    z_novo["name"] = z_ant.get("name") or z_novo["name"]
            st.session_state["zones"] = no_canvas
            _sincronizar_nomes_zonas(st.session_state["zones"])
            ui.flash(t("analysis.zones_saved", n=len(no_canvas)))
            st.rerun()
    if limpar:
        st.session_state.pop("zones", None)
        _sincronizar_nomes_zonas(None)
        st.rerun()

    if not _mesmas_zonas(no_canvas, zonas):
        st.warning(t("analysis.zones_unsaved", save=rotulo_salvar), icon="⚠️")

    if zonas:
        # O canvas acima já mostra as áreas sobre o frame — não repetir o preview.
        st.caption(t("analysis.zones_defined_rename", n=len(zonas)))
        for ini in range(0, len(zonas), 3):   # no máximo 3 por linha
            cols_nomes = st.columns(3)
            for j, z in enumerate(zonas[ini:ini + 3]):
                i = ini + j
                c_nome, c_del = cols_nomes[j].columns([5, 1], vertical_alignment="bottom")
                z["name"] = c_nome.text_input(
                    t("analysis.zone_n", n=i + 1), value=z["name"],
                    key=_chave_nome_zona(i))
                c_del.button("🗑️", key=f"zona_excluir_{i}", on_click=_excluir_zona,
                             args=(i,), help=t("analysis.zone_delete"))
    else:
        st.caption(t("analysis.no_zones_global"))


def _descartar_video_atual():
    """Apaga o vídeo enviado (arquivo temporário) e o seu estado na sessão."""
    caminho = st.session_state.pop("video_path", None)
    for k in ("video_token", "video_nome", "video_tamanho", "_keep_trecho"):
        st.session_state.pop(k, None)
    if caminho and os.path.exists(caminho):
        try:
            os.remove(caminho)
        except OSError:
            pass


def _ao_mudar_upload():
    # O usuário removeu o arquivo no próprio uploader (botão ✕). Trocar de tela
    # também esvazia o widget, mas NÃO dispara este callback — por isso o vídeo
    # continua disponível quando a pessoa volta para Nova Análise.
    if st.session_state.get("video_upload") is None:
        _descartar_video_atual()


def _limite_upload():
    """Tamanho máximo de upload configurado no servidor, legível ('3 GB')."""
    try:
        mb = float(st.get_option("server.maxUploadSize") or 200)
    except Exception:
        mb = 200.0
    return f"{mb / 1000:g} GB" if mb >= 1000 else f"{mb:g} MB"


def _onboarding_nova_analise():
    """Estado vazio (antes do upload): o fluxo em 4 passos + dicas de gravação."""
    passos = [
        ("📤", t("analysis.onb_upload_title"), t("analysis.onb_upload")),
        ("✂️", t("analysis.onb_trim_title"), t("analysis.onb_trim")),
        ("📏", t("analysis.onb_prepare_title"), t("analysis.onb_prepare")),
        ("▶️", t("analysis.onb_process_title"), t("analysis.onb_process")),
    ]
    for i, (col, (icone, titulo, texto)) in enumerate(zip(st.columns(4), passos), 1):
        with col, st.container(border=True):
            st.markdown(f"**{i} · {icone} {titulo}**")
            st.caption(texto)
    st.caption(t("analysis.onb_tips", max=_limite_upload(),
                 history=t("sidebar.tab_my_analyses")))


def _secao(titulo):
    """Seção numerada da Nova Análise (contêiner com borda + título)."""
    caixa = st.container(border=True)
    caixa.markdown(f"##### {titulo}")
    return caixa


def _chips_preparo():
    """Chips de estado da preparação: escala e áreas."""
    escala = st.session_state.get("pixels_per_mm", 0.0) or 0.0
    n_zonas = len(st.session_state.get("zones") or [])
    chips = [
        ui.chip(t("analysis.chip_scale", scale=escala), "ok") if escala > 0
        else ui.chip(t("analysis.chip_no_scale"), "warn"),
        ui.chip(t("analysis.chip_zones", n=n_zonas), "ok") if n_zonas
        else ui.chip(t("analysis.chip_no_zones"), "warn"),
    ]
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def _resumo_parametros(nome_modelo, frame_skip, conf, n_amostras, num_placas,
                       ampliar_areas, gerar_video, coletar_candidatos, limiar_revisao):
    """Resumo (somente leitura) do que será usado no processamento."""
    zonas = st.session_state.get("zones") or []
    escala = st.session_state.get("pixels_per_mm", 0.0) or 0.0
    sim, nao = t("analysis.sum_on"), t("analysis.sum_off")
    amostragem = t("analysis.sum_sampling_value", skip=frame_skip)
    if n_amostras:
        amostragem += " " + t("analysis.sum_sampling_est", n=n_amostras)
    alvos = (t("analysis.sum_areas_value", n=len(zonas)) if zonas
             else t("analysis.sum_dishes_value", n=num_placas))
    recorte = (sim if ampliar_areas else nao) if zonas else t("analysis.sum_crop_no_areas")
    candidatos = (t("analysis.sum_candidates_value", thr=limiar_revisao)
                  if coletar_candidatos else nao)
    itens = [
        (t("analysis.sum_model"), f"`{nome_modelo}`" if nome_modelo else "—"),
        (t("analysis.sum_sampling"), amostragem),
        (t("sidebar.conf_threshold"), f"{conf:.2f}"),
        (t("analysis.sum_targets"), alvos),
        (t("analysis.sum_scale"), (f"{escala:.2f} px/mm" if escala > 0
                                   else t("analysis.chip_no_scale"))),
        (t("analysis.sum_crop"), recorte),
        (t("analysis.sum_review_video"), sim if gerar_video else nao),
        (t("analysis.sum_candidates"), candidatos),
    ]
    metade = (len(itens) + 1) // 2
    for col, parte in zip(st.columns(2), (itens[:metade], itens[metade:])):
        col.markdown("\n".join(f"- {rot}: **{val}**" for rot, val in parte))
    if zonas:
        st.caption(t("analysis.zones_as_dishes", n=len(zonas)))
    st.caption(t("analysis.settings_hint", settings=t("sidebar.settings")))


def _cartao_pos_execucao(resumo):
    """Cartão com o resumo da última execução e atalhos para as outras telas."""
    with st.container(border=True):
        st.markdown(t("analysis.run_card_title", id=resumo["id"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("analysis.run_card_analysis"), f"#{resumo['id']}")
        c2.metric(t("analysis.run_card_detections"), resumo["detections"])
        c3.metric(t("analysis.run_card_dishes"), resumo["dishes"])
        c4.metric(t("analysis.run_card_time"), _fmt_duracao(resumo["secs"]))
        st.caption(t("analysis.run_card_below"))
        botoes = [("history", t("analysis.go_results"),
                   t("analysis.go_results_help", id=resumo["id"],
                     history=t("sidebar.tab_my_analyses")))]
        if resumo.get("video"):
            botoes.append(("review", t("analysis.go_review"), None))
        if resumo.get("candidates"):
            botoes.append(("annotate", t("analysis.go_annotate", n=resumo["candidates"]),
                           None))
        for col, (destino, rotulo, ajuda) in zip(st.columns(3), botoes):
            if col.button(rotulo, key=f"pos_run_{destino}", help=ajuda, width="stretch",
                          type="primary" if destino == "history" else "secondary"):
                if destino == "history":
                    # Abre o histórico já com esta análise selecionada.
                    st.session_state["_keep_hist_analise"] = resumo["id"]
                    st.session_state.pop("hist_analise_sel", None)
                go_to(destino)


def render_nova_analise(user_id, model, backend, class_names,
                        frame_skip, input_size, conf, nms, limiar_parada_px,
                        num_placas=1, ampliar_areas=False, gerar_video=False,
                        coletar_candidatos=False, limiar_revisao=0.5,
                        nome_modelo=None):
    st.session_state.setdefault("pixels_per_mm", 0.0)
    # O passo a passo fica acima do envio, mas só se sabe se há vídeo depois
    # de desenhar o uploader — por isso um contêiner reservado.
    topo = st.container()

    # ===== 1 · Vídeo e trecho =====
    sec_video = _secao(t("analysis.section_video"))
    with sec_video:
        uploaded = st.file_uploader(
            t("analysis.upload_video"), type=["mp4", "avi", "mov"],
            key="video_upload", on_change=_ao_mudar_upload,
        )
    caminho_salvo = st.session_state.get("video_path")
    if uploaded is None:
        # Widgets de telas não exibidas perdem o estado: o uploader volta vazio
        # ao retornar a esta tela, mas o vídeo enviado continua em disco.
        if caminho_salvo and os.path.exists(caminho_salvo):
            sec_video.caption(t("analysis.video_kept",
                                name=st.session_state.get("video_nome", "")))
        else:
            with topo:
                _onboarding_nova_analise()
            return

    # Salva o upload em disco por streaming, uma única vez por arquivo
    token = (uploaded.name, uploaded.size) if uploaded is not None else None
    if token is not None and st.session_state.get("video_token") != token:
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
        st.session_state["video_nome"] = uploaded.name
        st.session_state["video_tamanho"] = uploaded.size
        st.session_state.pop("_keep_trecho", None)
        st.session_state.pop("resultado", None)
        st.session_state.pop("resultado_resumo", None)
        log_event("video_uploaded", f"Upload de '{uploaded.name}'",
                  user_id=user_id, tamanho_bytes=uploaded.size)

    video_path = st.session_state["video_path"]
    video_nome = st.session_state.get("video_nome") or os.path.basename(video_path)
    video_tamanho = st.session_state.get("video_tamanho") or os.path.getsize(video_path)
    trecho_ok = True
    n_amostras = None
    with sec_video:
        # Altura/centralização do vídeo agora vêm do CSS global (ui.py).
        st.video(video_path)

        # Trecho do vídeo a analisar (início e fim em hh:mm:ss)
        dur = _duracao_video(video_path)
        if dur and dur >= 1:
            t_min, t_max = dtime(0, 0, 0), _seg_para_time(int(dur))
            # O trecho fica numa chave comum (não de widget) para sobreviver à troca
            # de tela; o widget é recriado a partir dela.
            trecho = st.session_state.get("_keep_trecho")
            if not (trecho and trecho[0] == video_path
                    and t_min <= trecho[1][0] <= trecho[1][1] <= t_max):
                trecho = (video_path, (t_min, t_max))
                # Vídeo novo: o slider (com key) ignoraria `value=` e manteria o
                # trecho do vídeo anterior quando as durações coincidem.
                if st.session_state.get("trecho_video", trecho[1]) != trecho[1]:
                    st.session_state["trecho_video"] = trecho[1]
            ini_t, fim_t = st.slider(
                t("analysis.segment"), min_value=t_min, max_value=t_max,
                value=trecho[1], step=dtimedelta(seconds=1), format="HH:mm:ss",
                help=t("analysis.segment_help"), key="trecho_video",
            )
            st.session_state["_keep_trecho"] = (video_path, (ini_t, fim_t))
            inicio_s = float(_time_para_seg(ini_t))
            # Mantendo o fim no máximo, processa até o fim real do vídeo.
            fim_s = None if fim_t == t_max else float(_time_para_seg(fim_t))
            fim_efetivo = dur if fim_s is None else fim_s
            if fim_efetivo <= inicio_s:
                trecho_ok = False
                st.warning(t("analysis.end_before_start"))
            else:
                fps_v = _fps_video(video_path)
                if fps_v > 0:
                    n_amostras = max(1, int(np.ceil(
                        (fim_efetivo - inicio_s) * fps_v / max(1, frame_skip))))
            resumo_trecho = t("analysis.segment_summary", start=_fmt_hms(inicio_s),
                              end=_fmt_hms(fim_efetivo), total=_fmt_hms(dur))
            if n_amostras:
                resumo_trecho += " " + t("analysis.segment_estimate",
                                         n=n_amostras, skip=frame_skip)
            st.caption(resumo_trecho)
        else:
            inicio_s, fim_s = 0.0, None

    # ===== 2 · Preparação (opcional) =====
    # Ferramentas de desenho (escala/áreas): apenas UMA fica ativa por vez.
    # Dois st_canvas na mesma página conflitam e às vezes não renderizam.
    with _secao(t("analysis.section_prep")):
        _chips_preparo()
        _ferramentas = {
            "none": t("analysis.tool_none"),
            "scale": t("analysis.tool_scale"),
            "zones": t("analysis.tool_zones"),
        }
        if st.session_state.get("ferramenta_desenho") not in (None, *_ferramentas):
            st.session_state.pop("ferramenta_desenho", None)  # valor antigo (rótulo)
        _ult_ferr = st.session_state.get("_keep_ferramenta", "none")
        ferramenta = st.segmented_control(
            t("analysis.tool"), list(_ferramentas), format_func=_ferramentas.get,
            key="ferramenta_desenho", required=True,
            default=_ult_ferr if _ult_ferr in _ferramentas else "none",
            help=t("analysis.tool_help"),
        )
        if ferramenta not in _ferramentas:
            ferramenta = "none"
        st.session_state["_keep_ferramenta"] = ferramenta
        if ferramenta == "scale":
            ferramenta_calibracao(video_path)
        elif ferramenta == "zones":
            ferramenta_zonas(video_path)

    # ===== 3 · Processar =====
    with _secao(t("analysis.section_process")):
        _resumo_parametros(nome_modelo, frame_skip, conf, n_amostras, num_placas,
                           ampliar_areas, gerar_video, coletar_candidatos,
                           limiar_revisao)
        falha = st.session_state.pop("_analise_falha", None)
        if falha:
            st.error(falha)
        processar = st.button(
            t("analysis.process_video"), type="primary", disabled=not trecho_ok,
            help=None if trecho_ok else t("analysis.process_disabled_help"),
            on_click=_marcar_ocupado,
        )
        st.caption(t("analysis.keep_page_open"))
        if processar:
            with _tarefa_longa():
                _executar_analise(user_id, model, backend, class_names, frame_skip,
                                  input_size, conf, nms, num_placas, ampliar_areas,
                                  gerar_video, coletar_candidatos, limiar_revisao,
                                  video_path, video_nome, video_tamanho, inicio_s,
                                  fim_s)

    resultado = st.session_state.get("resultado")
    if resultado is not None and resultado.primeiro_frame is not None:
        resumo = st.session_state.get("resultado_resumo")
        if resumo:
            _cartao_pos_execucao(resumo)
        exibir_resultados(
            resultado, st.session_state.get("resultado_frame_skip", frame_skip),
            limiar_parada_px,
            pixels_per_mm=st.session_state.get("resultado_pixels_per_mm"),
            zones=st.session_state.get("resultado_zones"),
            conf=st.session_state.get("resultado_conf", conf),
        )


def _executar_analise(user_id, model, backend, class_names, frame_skip, input_size,
                      conf, nms, num_placas, ampliar_areas, gerar_video,
                      coletar_candidatos, limiar_revisao, video_path, video_nome,
                      video_tamanho, inicio_s, fim_s):
    """Roda o processamento dentro de um st.status com as etapas e salva a análise."""
    zonas = st.session_state.get("zones") or []
    params = dict(frame_skip=frame_skip, input_size=input_size,
                  conf=conf, nms=nms, backend=backend,
                  pixels_per_mm=st.session_state.get("pixels_per_mm", 0.0) or None,
                  zones=json.dumps(zonas) if zonas else None)
    # Com áreas definidas, cada área é uma placa: o nº de abelhas = nº de áreas.
    n_alvos = len(zonas) if zonas else num_placas
    usar_ampliacao = bool(zonas) and ampliar_areas
    log_params = {k: v for k, v in params.items() if k != "zones"}
    log_event("analysis_started", f"Iniciando análise de '{video_nome}'",
              user_id=user_id, num_placas=n_alvos, usa_zonas=bool(zonas),
              **log_params)
    t0 = time.perf_counter()
    with st.status(t("analysis.processing", skip=frame_skip, n=n_alvos),
                   expanded=True) as caixa:
        def _fase(msg):
            caixa.update(label=msg)
            st.write(msg)

        try:
            resultado = processar_video(
                video_path, model, class_names, frame_skip, conf, nms,
                num_alvos=num_placas, zones=zonas or None,
                ampliar_areas=usar_ampliacao, gerar_video=gerar_video,
                inicio_s=inicio_s, fim_s=fim_s,
                coletar_candidatos=coletar_candidatos, limiar_revisao=limiar_revisao,
                fase=_fase,
            )
        except Exception as e:
            log_event("analysis_failed", f"Erro ao processar '{video_nome}': {e}",
                      user_id=user_id, level=logging.ERROR)
            caixa.update(label=t("analysis.status_failed"), state="error")
            # A página é redesenhada ao fim (_tarefa_longa): a mensagem vai
            # para a sessão para continuar visível.
            st.session_state["_analise_falha"] = t("analysis.processing_error", error=e)
            return
        except BaseException:
            # Execução interrompida pelo Streamlit (aba fechada, barra lateral
            # alterada...): nada foi salvo; registra e deixa a interrupção seguir.
            log_event("analysis_interrupted",
                      f"Análise de '{video_nome}' interrompida antes de salvar",
                      user_id=user_id, level=logging.WARNING)
            raise

        if resultado is None or resultado.primeiro_frame is None:
            caixa.update(label=t("analysis.status_failed"), state="error")
            st.session_state["_analise_falha"] = (
                t("core.video_open_error") if resultado is None
                else t("analysis.no_frames"))
            return

        _fase(t("analysis.phase_saving"))
        analysis_id = salvar_analise(
            user_id, video_nome, video_tamanho, params, resultado
        )
        st.write(t("analysis.phase_saved", id=analysis_id))
        st.session_state["resultado"] = resultado
        st.session_state["resultado_frame_skip"] = frame_skip
        st.session_state["resultado_pixels_per_mm"] = params["pixels_per_mm"]
        # Cópia: os nomes das áreas em Nova Análise são editados no próprio dict.
        st.session_state["resultado_zones"] = copy.deepcopy(zonas)
        st.session_state["resultado_conf"] = conf
        # Vídeo de conferência: remove o anterior e guarda o novo caminho.
        antigo_video = st.session_state.pop("video_deteccoes", None)
        if antigo_video and os.path.exists(antigo_video):
            try:
                os.remove(antigo_video)
            except OSError:
                pass
        st.session_state["video_deteccoes"] = resultado.video_deteccoes
        st.session_state["video_deteccoes_nome"] = video_nome
        # Salva os candidatos para reanotação (se coletados).
        n_cand = 0
        for cand in (resultado.candidatos or []):
            database.add_candidate(
                user_id, analysis_id, video_nome, cand["frame_index"],
                cand["reason"], cand["pred_conf"], cand["pred_boxes"],
                cand["width"], cand["height"], cand["image_jpg"],
                area=cand.get("area"))
            n_cand += 1
        if n_cand:
            st.write(t("analysis.phase_candidates", n=n_cand))
            log_event("candidates_collected", f"{n_cand} candidatos de '{video_nome}'",
                      user_id=user_id, analysis_id=analysis_id)
        # Contexto p/ anotar, na Conferência, os frames abaixo do limiar.
        st.session_state["conf_ctx"] = _contexto_conferencia(
            resultado, analysis_id, video_path, video_nome, frame_skip, conf, nms,
            zonas, usar_ampliacao)
        decorrido = time.perf_counter() - t0
        caixa.update(label=t("analysis.status_done", id=analysis_id,
                             time=_fmt_duracao(decorrido)),
                     state="complete", expanded=False)

    st.session_state["resultado_resumo"] = {
        "id": analysis_id, "detections": int(len(resultado.centroides)),
        "dishes": int(resultado.num_alvos), "secs": decorrido,
        "video": bool(resultado.video_deteccoes), "candidates": n_cand,
    }
    # Toast via flash: a página é redesenhada logo depois (_tarefa_longa).
    ui.flash(t("analysis.toast_saved", id=analysis_id))
    if class_names and TARGET_CLASS not in class_names:
        ui.flash(t("core.class_not_found", cls=TARGET_CLASS, names=YOLO_NAMES), "warning")


def render_conferencia(user_id, model, class_names):
    """Aba de conferência: reproduz o vídeo anotado das detecções do último
    processamento e permite anotar os frames que ficaram abaixo do limiar."""
    st.markdown(t("analysis.review_title"))
    st.caption(t("analysis.review_caption"))
    path = st.session_state.get("video_deteccoes")
    tem_video = bool(path and os.path.exists(path))
    ctx = st.session_state.get("conf_ctx")
    if not isinstance(ctx, dict) or not database.analysis_exists(
            ctx.get("analysis_id"), user_id):
        # Sem contexto, ou a análise foi excluída no Histórico.
        st.session_state.pop("conf_ctx", None)
        ctx = None
    if not tem_video and ctx is None:
        _conferencia_vazia()
        return
    if tem_video:
        _conferencia_video(path)
    else:
        st.info(t("review.no_video_but_ctx", option=t("sidebar.review_video")))
    if ctx is not None:
        _conf_secao_anotacao(user_id, model, class_names, ctx)


def _conferencia_vazia():
    """Estado vazio: como gerar o vídeo (rótulos exatos da barra lateral)."""
    with st.container(border=True):
        st.markdown(t("analysis.review_empty_title"))
        st.markdown(t("analysis.review_empty_steps",
                      settings=t("sidebar.settings"),
                      section=t("sidebar.detection"),
                      option=t("sidebar.review_video"),
                      new=t("sidebar.tab_new_analysis")))
        if st.session_state.get("cfg_gerar_video") is False:
            st.warning(t("analysis.review_option_off", option=t("sidebar.review_video")))
        st.caption(t("analysis.review_empty_note"))
        if st.button(t("analysis.go_new"), type="primary",
                     key="conf_ir_nova"):
            go_to("new")


def _conferencia_video(path):
    """Vídeo de conferência (legenda + player + download)."""
    nome = st.session_state.get("video_deteccoes_nome")
    if nome:
        st.caption(t("analysis.review_source", name=_md_esc(nome)))
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;"
        f"margin:2px 0 8px'>"
        f"<span style='font-size:.85rem;opacity:.8'>{ui.esc(t('analysis.review_legend_title'))}</span>"
        f"{ui.chip(ui.esc(t('analysis.review_legend_areas')), 'warn')}"
        f"{ui.chip(ui.esc(t('analysis.review_legend_raw')), 'raw')}"
        f"{ui.chip(ui.esc(t('analysis.review_legend_accepted')), 'ok')}</div>",
        unsafe_allow_html=True)
    st.video(path)
    aid = (st.session_state.get("resultado_resumo") or {}).get("id")
    nome_arq = (t("analysis.review_file", id=aid) if aid is not None
                else t("analysis.review_file_noid"))
    try:
        with open(path, "rb") as fh:
            st.download_button(t("analysis.review_download"), data=fh.read(),
                               file_name=nome_arq, mime="video/mp4", on_click="ignore")
    except OSError:
        st.warning(t("analysis.review_unreadable"))


# --- Conferência: anotar os frames que ficaram abaixo do limiar ---

def _contexto_conferencia(resultado, analysis_id, video_path, video_nome, frame_skip,
                          conf, nms, zonas, ampliar_areas) -> dict:
    """O que a Conferência precisa para refazer, depois, os frames/áreas sem
    detecção aceita do último processamento (st.session_state["conf_ctx"])."""
    return {
        "analysis_id": int(analysis_id), "video_path": video_path,
        "video_nome": video_nome, "frame_skip": int(frame_skip),
        "conf": float(conf), "nms": float(nms), "zones": _zonas_congeladas(zonas),
        "ampliar_areas": bool(ampliar_areas),
        "num_alvos": int(resultado.num_alvos),
        "frames_lidos": list(resultado.frames_lidos or []),
        "aceitos": conferencia.frames_aceitos(resultado.indices_frames,
                                              resultado.track_ids),
    }


def _zonas_congeladas(zonas) -> list:
    """Áreas válidas com o nome JÁ resolvido (cópia independente da sessão).

    Os nomes são a chave dos candidatos no banco (frame, área): renomear a área
    em Nova Análise ou trocar o idioma (nome padrão "Área n") depois do
    processamento não pode mudá-los. Nomes repetidos ganham um sufixo " (n)"
    para que cada área tenha a sua chave.
    """
    polys, nomes = _zonas_validas(zonas)
    saida, vistos = [], set()
    for i, (p, nome) in enumerate(zip(polys, nomes), 1):
        if nome in vistos:
            nome = f"{nome} ({i})"
        vistos.add(nome)
        saida.append({"name": nome, "points": p.tolist()})
    return saida


def _chave_par(par, nomes):
    """(frame, nome da área|None): a chave de um par como fica no banco."""
    f, zi = par
    return f, (nomes[zi] if zi is not None else None)


def _pares_conferencia(ctx):
    """(pares (frame, área|None) sem detecção aceita, polígonos, nomes das áreas)."""
    polys, nomes = _zonas_validas(ctx.get("zones"))
    n_alvos = len(polys) if polys else ctx.get("num_alvos", 1)
    pares = conferencia.pares_sem_deteccao(ctx.get("frames_lidos"), ctx.get("aceitos"),
                                           n_alvos, bool(polys))
    return pares, polys, nomes


def _melhor_abaixo(sugeridas, conf) -> float:
    """Melhor score das sugestões ABAIXO do limiar da análise (0 = nenhuma).

    Sem áreas e com 2+ abelhas, o frame tem detecções aceitas (≥ limiar) das
    outras abelhas; com áreas, o passe no recorte pode achar a abelha acima do
    limiar. Essas caixas ficam nas sugestões (para validar), mas não definem o
    motivo 'baixa_confianca' nem a confiança mostrada.
    """
    return max((b[4] for b in sugeridas if b[4] < conf), default=0.0)


def _candidato_conferencia(frame, indice, zi, model, conf_captura, conf, nms, id_alvo,
                           polys, paths, nomes):
    """Candidato de um frame: recorte da área `zi` ou o frame inteiro (zi None)."""
    if zi is None:
        sugeridas = _caixas_alvo(model, frame, conf_captura, nms, id_alvo)
        return _montar_candidato(indice, frame, _melhor_abaixo(sugeridas, conf),
                                 sugeridas)
    outras = [p for j, p in enumerate(paths) if j != zi]
    rc = _recorte_candidato(frame, model, conf_captura, nms, id_alvo,
                            polys[zi], paths[zi], outras)
    if rc is None:
        return None
    recorte, _, sugeridas = rc
    return _montar_candidato(indice, recorte, _melhor_abaixo(sugeridas, conf),
                             sugeridas, nomes[zi])


def coletar_frames_abaixo_limiar(ctx, model, class_names, conf_captura, max_itens,
                                 progresso=None, ignorar=None, falhas=None):
    """Frames/áreas do último processamento sem detecção aceita → candidatos.

    Escolhe até `max_itens` pares (frame, área) espalhados pelo trecho, lê cada
    frame uma vez e roda um passe de baixa confiança (`conf_captura`) para as
    sugestões. Com áreas, cada item é o recorte da área (caixas em px do
    recorte); sem áreas, o frame inteiro. `ignorar`: {(frame, área)} já
    guardados. `falhas`: set opcional que recebe as chaves (frame, área) que
    não puderam ser lidas/recortadas. `progresso(i, n)`: chamado a cada frame
    lido. Devolve dicts no mesmo formato dos candidatos de processar_video.
    """
    from matplotlib.path import Path as _MplPath
    pares, polys, nomes = _pares_conferencia(ctx)
    ignorar = ignorar or set()
    falhas = falhas if falhas is not None else set()
    pares = [par for par in pares if _chave_par(par, nomes) not in ignorar]
    por_frame = {}
    for f, zi in conferencia.espalhar_no_tempo(pares, ctx.get("frames_lidos"),
                                               max_itens):
        por_frame.setdefault(f, []).append(zi)
    paths = [_MplPath(p) for p in polys]
    id_alvo = _id_classe_alvo(class_names)
    nms = float(ctx.get("nms", 0.3))
    conf = float(ctx.get("conf", 1.0))
    saida = []
    cap = cv2.VideoCapture(ctx["video_path"])
    try:
        if not cap.isOpened():
            raise OSError(t("core.video_open_error"))
        frames = sorted(por_frame)
        for i, f in enumerate(frames, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            for zi in por_frame[f]:
                cand = (_candidato_conferencia(frame, f, zi, model, conf_captura, conf,
                                               nms, id_alvo, polys, paths, nomes)
                        if ok and frame is not None else None)
                if cand is not None:
                    saida.append(cand)
                else:
                    falhas.add(_chave_par((f, zi), nomes))
            if progresso is not None:
                progresso(i, len(frames))
    finally:
        cap.release()
    return saida


def _conf_coletar(user_id, model, class_names, ctx, conf_captura, max_itens):
    """Trabalho longo do botão: coleta e guarda os candidatos (origem Conferência)."""
    aid = ctx["analysis_id"]
    with st.status(t("review.collecting"), expanded=True) as caixa:
        barra = st.progress(0.0)

        def _prog(i, n):
            barra.progress(i / max(n, 1), text=t("review.collect_progress", i=i, n=n))

        # Pares que não puderam ser lidos/recortados ficam no contexto: não
        # voltam a ocupar vagas nem contam como "faltando coletar".
        falhas = ctx.setdefault("falhas", set())
        try:
            itens = coletar_frames_abaixo_limiar(
                ctx, model, class_names, conf_captura, max_itens, progresso=_prog,
                ignorar=database.candidate_keys(user_id, aid) | falhas,
                falhas=falhas)
        except Exception as e:
            log_event("candidates_collect_failed",
                      f"Erro ao coletar frames na conferência: {e}",
                      user_id=user_id, analysis_id=aid, level=logging.ERROR)
            caixa.update(label=t("review.collect_failed_status"), state="error")
            ui.flash(t("review.collect_failed", error=e), "error")
            return
        n = 0
        for c in itens:
            if database.candidate_exists(user_id, aid, c["frame_index"], c.get("area")):
                continue
            database.add_candidate(
                user_id, aid, ctx.get("video_nome"), c["frame_index"], c["reason"],
                c["pred_conf"], c["pred_boxes"], c["width"], c["height"],
                c["image_jpg"], area=c.get("area"), origin="conferencia")
            n += 1
        caixa.update(label=t("review.collect_done_status", n=n), state="complete",
                     expanded=False)
    log_event("candidates_collected_review",
              f"{n} frame(s) abaixo do limiar coletado(s) na conferência de "
              f"'{ctx.get('video_nome')}'",
              user_id=user_id, analysis_id=aid, conf_captura=conf_captura)
    if n:
        ui.flash(t("review.collected", n=n))
    else:
        ui.flash(t("review.collected_none"), "info")


def _conf_secao_anotacao(user_id, model, class_names, ctx):
    """Seção da Conferência: coletar e anotar os frames sem detecção aceita."""
    aid = ctx["analysis_id"]
    conf = float(ctx.get("conf", 0.4))
    pares, polys, nomes = _pares_conferencia(ctx)
    n_frames, n_pares = conferencia.resumo_faltantes(pares)
    n_lidos = len(ctx.get("frames_lidos") or [])
    with st.container(border=True):
        st.markdown(t("review.annot_title"))
        st.caption(t("review.annot_caption", id=aid))
        if polys:
            st.markdown(t("review.stats_zones", frames=n_frames, total=n_lidos,
                          pairs=n_pares, conf=conf))
            st.caption(t("review.mode_zones", n=len(polys)))
        else:
            st.markdown(t("review.stats_whole", frames=n_frames, total=n_lidos,
                          n=ctx.get("num_alvos", 1), conf=conf))
            st.caption(t("review.mode_whole"))
        ja = database.candidate_keys(user_id, aid)
        falhas = ctx.get("falhas") or set()
        chaves = [_chave_par(par, nomes) for par in pares]
        n_ja = sum(1 for k in chaves if k in ja)
        n_falhas = sum(1 for k in chaves if k in falhas and k not in ja)
        restantes = n_pares - n_ja - n_falhas
        if n_falhas:
            st.caption(t("review.unreadable", n=n_falhas))
        pedido = _conf_controles_coleta(ctx, n_pares, n_ja, restantes,
                                        model is not None)
        if pedido is not None:
            # Trabalho longo: um elemento simples no lugar do bloco do anotador
            # tira da tela (e de alcance) o anotador/botões da execução anterior,
            # que continuariam clicáveis e interromperiam a coleta.
            st.empty()
            with _tarefa_longa():
                _conf_coletar(user_id, model, class_names, ctx, *pedido)
        with st.container():
            _conf_anotacao_analise(user_id, aid)


def _conf_controles_coleta(ctx, n_pares, n_ja, restantes, tem_modelo):
    """Quantidade, confiança das sugestões e o botão de coleta.

    Devolve (conf_captura, max_itens) quando o botão foi clicado; senão None.
    """
    aid = ctx["analysis_id"]
    conf = float(ctx.get("conf", 0.4))
    if n_pares == 0:
        st.success(t("review.none_missing", conf=conf))
        return None
    if n_ja:
        st.caption(t("review.already_collected", n=n_ja, left=restantes))
    if restantes == 0:
        st.info(t("review.all_collected"))
        return None
    if not os.path.exists(ctx.get("video_path") or ""):
        st.info(t("review.video_gone"))
        return None
    if not tem_modelo:
        st.info(t("review.needs_model"))
        return None
    max_itens = st.slider(t("review.max_items"), 5, 200, 50, step=5,
                          key="conf_col_max", help=t("review.max_items_help"))
    with st.expander(t("review.advanced")):
        teto = max(0.02, round(conf - 0.01, 2))
        padrao = min(teto, max(0.01, round(min(0.15, conf / 2), 2)))
        # A chave leva o id da análise: o teto muda com o limiar de cada análise.
        conf_captura = st.slider(t("review.suggest_conf"), 0.01, teto, padrao, 0.01,
                                 key=f"conf_col_capt_{aid}",
                                 help=t("review.suggest_conf_help", conf=conf))
    coletar = st.button(t("review.collect_btn"), type="primary", key="conf_col_coletar",
                        on_click=_marcar_ocupado)
    st.caption(t("review.collect_hint", n=min(max_itens, restantes)))
    return (conf_captura, max_itens) if coletar else None


@st.fragment
def _conf_anotacao_analise(user_id, aid):
    """Anotador restrito aos candidatos da análise + atalho para o treino.

    Fragmento: cada traço no canvas redesenha só este bloco, não o vídeo de
    conferência acima (que seria lido do disco de novo a cada interação).
    Salvar/validar/navegar chamam st.rerun() e redesenham a página toda.
    """
    b = database.candidate_breakdown(user_id, analysis_id=aid)
    if not sum(b[k] for k in ("pendente", "validado", "reanotado", "sem_abelha",
                              "descartado")):
        return
    st.divider()
    c_txt, c_btn = st.columns([3, 1], vertical_alignment="center")
    c_txt.markdown(t("review.counts", pending=b["pendente"], validated=b["validado"],
                     reannotated=b["reanotado"], no_bee=b["sem_abelha"],
                     discarded=b["descartado"]))
    if c_btn.button(t("review.go_train"), key="conf_ir_treinar", width="stretch"):
        st.session_state["_keep_anot_secao"] = "train"
        st.session_state.pop("anot_secao", None)
        go_to("annotate")
    _anotador(user_id, "rev_anot", analysis_id=aid)
def _box_para_rect_canvas(box, escala):
    """Caixa [x,y,w,h] (px imagem) -> objeto fabric rect (coords do canvas)."""
    x, y, w, h = box[:4]
    return {"type": "Rect", "left": x / escala, "top": y / escala,
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
    arquivo. Mescla com o que já foi importado. Devolve um dict com o que veio
    no .zip: imagens, rotulos, com_rotulo (imagens com .txt de mesmo nome),
    sem_rotulo (imagens sem .txt: entram no treino como imagens sem abelha) e
    rotulos_orfaos (.txt sem imagem: não são usados).
    """
    import io as _io
    import zipfile
    img_dir, lbl_dir = _externo_img_dir(), _externo_lbl_dir()
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stems_img, stems_lbl = set(), set()
    n_i = n_l = 0
    with zipfile.ZipFile(_io.BytesIO(conteudo_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            nome = os.path.basename(info.filename)
            if not nome or nome.startswith("."):
                continue
            stem, ext = os.path.splitext(nome)
            ext = ext.lower()
            if ext in _IMG_EXTS:
                with zf.open(info) as src, open(img_dir / nome, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n_i += 1
                stems_img.add(stem)
            elif ext == ".txt" and nome.lower() not in ("classes.txt", "readme.txt"):
                with zf.open(info) as src, open(lbl_dir / nome, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n_l += 1
                stems_lbl.add(stem)
    return {"imagens": n_i, "rotulos": n_l,
            "com_rotulo": len(stems_img & stems_lbl),
            "sem_rotulo": len(stems_img - stems_lbl),
            "rotulos_orfaos": len(stems_lbl - stems_img)}


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
            erros.append(t("model.err_unsupported_ext", name=nome))
            continue
        destino = _TRAINED_DIR / nome
        if destino.exists() and destino not in gravados:
            erros.append(t("model.replaced_existing", name=nome))
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
                erros.append(t("model.err_invalid_pt", name=p.name, msg=msg))
        elif ext == ".weights":
            cfg_par = p.with_suffix(".cfg")
            cfg = str(cfg_par) if cfg_par.exists() else YOLO_CFG
            try:
                cv2.dnn.readNetFromDarknet(cfg, str(p))
                salvos.append(f"{p.name} (cfg: {os.path.basename(cfg)})")
            except Exception as e:
                msg = (str(e).strip().splitlines() or [repr(e)])[-1][:90]
                if not cfg_par.exists():
                    erros.append(t("model.err_missing_cfg", name=p.name, stem=p.stem))
                    salvos.append(p.name)  # mantém; falta só o cfg
                else:
                    p.unlink(missing_ok=True)
                    erros.append(t("model.err_invalid_darknet", name=p.name, msg=msg))
    return salvos, erros


class SemImagensParaTreino(RuntimeError):
    """Nenhuma imagem rotulada utilizável para montar o dataset de treino."""


# Perfis de treino (Anotação > Treinar): modelo base, épocas e resolução.
PERFIS_TREINO = {
    "fast": {"base": "yolo11n.pt", "epochs": 30, "imgsz": 416},
    "balanced": {"base": "yolo11n.pt", "epochs": 100, "imgsz": 640},
    "accurate": {"base": "yolo11s.pt", "epochs": 200, "imgsz": 640},
}
MODELOS_BASE = ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"]


def dispositivo_treino() -> tuple[str, str]:
    """("gpu", nome da placa) se o PyTorch vê CUDA, senão ("cpu", "")."""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                return "gpu", torch.cuda.get_device_name(0)
            except Exception:
                return "gpu", ""
    except Exception:
        pass
    return "cpu", ""


def erro_treino_amigavel(e: BaseException) -> str:
    """Classifica uma falha de treino: "offline", "no_images", "oom" ou "other"."""
    if isinstance(e, SemImagensParaTreino):
        return "no_images"
    txt = f"{type(e).__name__}: {e}".lower()
    if isinstance(e, MemoryError) or any(k in txt for k in (
            "out of memory", "cannot allocate memory", "defaultcpuallocator",
            "not enough memory", "std::bad_alloc")):
        return "oom"
    if any(k in txt for k in (
            "urlopen", "connectionerror", "connection error", "name resolution",
            "name or service not known", "network is unreachable", "getaddrinfo",
            "max retries exceeded", "failed to download", "download failure",
            "httpsconnection", "timed out", "temporary failure")):
        return "offline"
    if any(k in txt for k in ("no images found", "no labels found",
                              "images not found", "dataset images not found")):
        return "no_images"
    return "other"


def treinar_modelo(user_id, base_model, epochs, imgsz, barra, status=None):
    """Fine-tuning Ultralytics a partir dos candidatos anotados do usuário.

    Monta o dataset YOLO em disco (candidatos + dataset externo importado),
    treina (GPU se disponível) e salva o melhor `.pt` em _TRAINED_DIR (volume de
    dados, persistente). Devolve (caminho_pt, n_imagens, n_caixas).
    `barra` é um st.progress (o texto mostra época, tempo e estimativa);
    `status`, se dado, é um st.empty que recebe o mesmo texto.
    """
    from ultralytics import YOLO
    import torch

    anotados = database.list_annotated_candidates(user_id)
    n_ext = contar_externo()[0]
    if not anotados and n_ext == 0:
        raise SemImagensParaTreino("no annotated images")
    tmp = tempfile.mkdtemp(prefix="bee_ds_")
    extra = [str(_externo_img_dir())] if n_ext > 0 else None
    data_yaml, n_imgs, n_caixas = dataset.export_to_dir(anotados, tmp,
                                                        extra_train_dirs=extra)
    if n_imgs + n_ext == 0:
        raise SemImagensParaTreino("no annotated images")

    device = 0 if torch.cuda.is_available() else "cpu"
    _TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    proj = str(_TRAINED_DIR / "_runs")
    nome_run = datetime.now().strftime("ft_%Y%m%d_%H%M%S")

    modelo = YOLO(base_model)
    inicio = time.time()

    def _cb(trainer):
        ep = int(getattr(trainer, "epoch", 0)) + 1
        total = max(int(epochs), 1)
        decorrido = time.time() - inicio
        restante = decorrido / ep * max(total - ep, 0)
        texto = t("model.train_epoch_progress", ep=ep, epochs=total,
                  elapsed=_fmt_duracao(decorrido), eta=_fmt_duracao(restante))
        try:
            barra.progress(min(ep / total, 1.0), text=texto)
            if status is not None:
                status.text(texto)
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


# --- Anotação: ciclo reanotar -> treinar -> dados ---

def _n_modelos_treinados() -> int:
    """Quantos modelos foram treinados no app (beemove_*.pt na pasta de modelos)."""
    try:
        return sum(1 for _ in _TRAINED_DIR.glob("beemove_*.pt"))
    except Exception:
        return 0


def _assinatura_dataset(anotados) -> tuple:
    """Muda sempre que as anotações mudam: nº de itens, maior id e conteúdo."""
    if not anotados:
        return (0, 0, 0)
    conteudo = json.dumps([(c["id"], c.get("status"), c.get("labels"))
                           for c in anotados])
    return (len(anotados), max(c["id"] for c in anotados),
            zlib.crc32(conteudo.encode()))


def _proximo_pendente(lista, atual_id):
    """Id do próximo candidato pendente depois de `atual_id` (circular) ou None."""
    ids = [c["id"] for c in lista]
    i = ids.index(atual_id) if atual_id in ids else -1
    for k in range(1, len(lista) + 1):
        c = lista[(i + k) % len(lista)]
        if c["id"] != atual_id and c.get("status") == "pendente":
            return c["id"]
    return None


def render_anotacao(user_id):
    """Ciclo de melhoria do modelo: reanotar candidatos, treinar e dados."""
    ui.header(t("annot.title"), t("annot.subtitle"))
    cont = database.candidate_counts(user_id)
    pend = cont.get("pendente", 0)
    anot = cont.get("anotado", 0) + cont.get("sem_abelha", 0)
    desc = cont.get("descartado", 0)
    coletados = pend + anot + desc
    n_ext = contar_externo()[0]
    st.markdown(t("annot.progress_line", collected=coletados, pending=pend,
                  annotated=anot, discarded=desc, trained=_n_modelos_treinados()))

    secoes = {"label": t("annot.section_label", n=pend),
              "train": t("annot.section_train"),
              "data": t("annot.section_data")}
    padrao = "label" if pend > 0 or (anot == 0 and n_ext == 0) else "train"
    if st.session_state.get("anot_secao") not in (None, *secoes):
        st.session_state.pop("anot_secao", None)
    ult = st.session_state.get("_keep_anot_secao")
    secao = st.segmented_control(
        t("annot.section"), list(secoes), format_func=secoes.get,
        key="anot_secao", required=True,
        default=ult if ult in secoes else padrao, label_visibility="collapsed")
    if secao not in secoes:
        secao = padrao
    st.session_state["_keep_anot_secao"] = secao

    if secao == "label":
        if coletados == 0:
            st.info(t("annot.no_candidates"))
        else:
            _anotador(user_id, "anot")
    elif secao == "train":
        _anot_treinar(user_id, anot, n_ext)
    else:
        _anot_dados(user_id)


def _rotulo_candidato(c):
    """Texto do candidato no seletor 'Ir para'."""
    area = f" · {c['area']}" if c.get("area") else ""
    conf = (t("annot.position_conf", conf=c["pred_conf"])
            if c.get("pred_conf") is not None else "")
    return t("annot.candidate_label", id=c["id"], area=area,
             frame=c["frame_index"], reason=t(f"annot.reason_{c['reason']}"),
             conf=conf, status=_status_candidato(c))


def _status_candidato(c):
    """Situação legível; anotados dizem se a predição foi validada ou reanotada."""
    txt = t(f"annot.status_{c['status']}")
    if c.get("status") == "anotado" and c.get("label_source") in ("validado",
                                                                   "reanotado"):
        txt = t(f"annot.source_{c['label_source']}")
    return txt


def _anot_escolher(prefixo, lista, coluna):
    """Selectbox 'Ir para' + pedido pendente de navegação; devolve o id escolhido."""
    rotulos = {c["id"]: _rotulo_candidato(c) for c in lista}
    ids = list(rotulos)
    # Pedido de navegação (anterior/próximo/avanço após salvar) feito na
    # execução anterior: aplica antes de desenhar o seletor.
    alvo = st.session_state.pop(f"_{prefixo}_goto", None)
    for candidato in (alvo, st.session_state.get(f"{prefixo}_sel"),
                      st.session_state.get(f"_keep_{prefixo}_sel")):
        if candidato in ids:
            alvo = candidato
            break
    else:
        alvo = ids[0]
    st.session_state[f"{prefixo}_sel"] = alvo
    escolhido = coluna.selectbox(t("annot.jump_to"), ids, key=f"{prefixo}_sel",
                                 format_func=lambda i: rotulos.get(i, str(i)))
    st.session_state[f"_keep_{prefixo}_sel"] = escolhido
    return escolhido


def _anotador(user_id, prefixo, analysis_id=None):
    """Reanotação: um candidato por vez, com anterior/próximo e avanço automático.

    Componente usado na Anotação (prefixo "anot", todos os candidatos) e na
    Conferência (só os de `analysis_id`). Todas as chaves de widget/sessão levam
    o `prefixo`, para as duas instâncias não colidirem.
    """
    try:
        _patch_canvas_compat()
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception:
        st.warning(t("annot.no_canvas"))
        return

    _filtros = ["pendente", "todos"]
    _ult_f = st.session_state.get(f"_keep_{prefixo}_filtro")
    c_filtro, c_ir = st.columns([1, 2], vertical_alignment="bottom")
    filtro = c_filtro.segmented_control(
        t("annot.show"), _filtros, key=f"{prefixo}_filtro", required=True,
        default=_ult_f if _ult_f in _filtros else "pendente",
        format_func={v: t(f"annot.filter_{v}") for v in _filtros}.get)
    if filtro not in _filtros:
        filtro = "pendente"
    st.session_state[f"_keep_{prefixo}_filtro"] = filtro
    lista = database.list_candidates(
        user_id, status="pendente" if filtro == "pendente" else None,
        analysis_id=analysis_id)
    if not lista:
        st.success(t("annot.all_done"))
        return

    escolhido = _anot_escolher(prefixo, lista, c_ir)
    ids = [c["id"] for c in lista]
    pos = ids.index(escolhido)
    info = lista[pos]

    c_ant, c_txt, c_prox = st.columns([1, 4, 1], vertical_alignment="center")
    if c_ant.button(t("annot.prev_btn"), key=f"{prefixo}_prev", width="stretch",
                    disabled=pos == 0):
        st.session_state[f"_{prefixo}_goto"] = ids[pos - 1]
        st.rerun()
    if c_prox.button(t("annot.next_btn"), key=f"{prefixo}_next", width="stretch",
                     disabled=pos >= len(ids) - 1):
        st.session_state[f"_{prefixo}_goto"] = ids[pos + 1]
        st.rerun()
    area = t("annot.position_area", area=_md_esc(info["area"])) if info.get("area") else ""
    c_txt.markdown(t("annot.position", i=pos + 1, n=len(ids),
                     video=_md_esc(info.get("video_name") or "—"), area=area,
                     frame=info["frame_index"]))
    conf = (t("annot.position_conf", conf=info["pred_conf"])
            if info.get("pred_conf") is not None else "")
    c_txt.caption(t("annot.position_detail",
                    reason=t(f"annot.reason_{info['reason']}"), conf=conf,
                    status=_status_candidato(info)))

    cand = database.get_candidate(escolhido, user_id)
    if not cand or not cand.get("image_jpg"):
        st.warning(t("annot.no_image"))
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
    brutas = json.loads(cand["pred_boxes"]) if cand.get("pred_boxes") else []
    sugeridas = [b[:4] for b in brutas]
    # Confiança mostrada = a da melhor caixa sugerida (não o pred_conf, que na
    # coleta do processamento é o score da detecção aceita da área).
    melhor_sug = max((float(b[4]) for b in brutas if len(b) > 4), default=None)
    base = json.loads(cand["labels"]) if cand.get("labels") else None
    if not base:
        base = sugeridas
    init = {"version": "4.4.0",
            "objects": [_box_para_rect_canvas(b, escala) for b in (base or [])]}

    st.markdown(t("annot.draw_instructions"))
    col_canvas, col_acoes = st.columns([3, 1])
    with col_canvas:
        canvas = st_canvas(
            fill_color="rgba(0,224,164,0.15)", stroke_width=2, stroke_color="#00E0A4",
            background_image=pil, update_streamlit=True, initial_drawing=init,
            height=canvas_h, width=canvas_w, drawing_mode="rect",
            key=f"canvas_{prefixo}_{cand['id']}",
        )
    with col_acoes:
        _anot_acoes(user_id, prefixo, lista, cand, canvas, escala, sugeridas,
                    melhor_sug, base or [], largura, altura)


def _caixas_do_canvas(canvas, escala):
    """Caixas [x,y,w,h] (px imagem) desenhadas no canvas; None se o canvas
    ainda não devolveu nada (antes da primeira interação)."""
    dados = canvas.json_data if canvas is not None else None
    if not dados or dados.get("objects") is None:
        return None
    caixas = [_rect_para_box(o, escala) for o in dados["objects"]
              if str(o.get("type", "")).lower() == "rect" and o.get("width")]
    return [b for b in caixas if b[2] > 1 and b[3] > 1]


def _mesmas_caixas(a, b, tol=2.0) -> bool:
    """Mesmo conjunto de caixas (na mesma ordem), com tolerância em px."""
    return len(a) == len(b) and all(
        abs(float(p) - float(q)) <= tol for ca, cb in zip(a, b)
        for p, q in zip(ca[:4], cb[:4]))


def _anot_acoes(user_id, prefixo, lista, cand, canvas, escala, sugeridas,
                melhor_sug, iniciais, largura, altura):
    """Botões do anotador: validar a predição, salvar caixas, sem abelha, descartar.

    Validar grava as caixas SUGERIDAS como estão; por isso só fica disponível
    (e como botão principal) enquanto o canvas mostra exatamente as sugestões —
    sem edições e sem rótulos anteriores diferentes delas.
    """
    def _concluir(msg, tipo="success"):
        """Depois de validar/salvar/marcar/descartar, abre o próximo pendente."""
        ui.flash(msg, tipo)
        prox = _proximo_pendente(lista, cand["id"])
        if prox is not None:
            st.session_state[f"_{prefixo}_goto"] = prox
        st.rerun()

    st.caption(t("annot.frame_info", w=largura, h=altura))
    if sugeridas:
        st.caption(t("annot.prediction_info", n=len(sugeridas),
                     conf=melhor_sug if melhor_sug is not None
                     else float(cand.get("pred_conf") or 0)))
    no_canvas = _caixas_do_canvas(canvas, escala)
    if no_canvas is None:
        no_canvas = iniciais
    pode_validar = bool(sugeridas) and _mesmas_caixas(no_canvas, sugeridas)
    if not sugeridas:
        ajuda_validar = t("annot.validate_disabled_help")
    elif not pode_validar:
        ajuda_validar = t("annot.validate_edited_help")
    else:
        ajuda_validar = t("annot.validate_help")
    if st.button(t("annot.validate_btn"), type="primary" if pode_validar else "secondary",
                 width="stretch", key=f"{prefixo}_validar", disabled=not pode_validar,
                 help=ajuda_validar):
        database.set_candidate_labels(cand["id"], user_id, sugeridas, "anotado",
                                      source="validado")
        log_event("candidate_validated",
                  f"Candidato #{cand['id']}: predição validada ({len(sugeridas)} caixa(s))",
                  user_id=user_id, analysis_id=cand.get("analysis_id"))
        _concluir(t("annot.flash_validated", id=cand["id"], n=len(sugeridas)))
    if st.button(t("annot.save_btn"), type="secondary" if pode_validar else "primary",
                 width="stretch", key=f"{prefixo}_salvar"):
        caixas = _caixas_do_canvas(canvas, escala) or []
        if not caixas:
            st.warning(t("annot.no_boxes"))
        else:
            database.set_candidate_labels(cand["id"], user_id, caixas, "anotado",
                                          source="reanotado")
            log_event("candidate_reannotated",
                      f"Candidato #{cand['id']}: reanotado ({len(caixas)} caixa(s))",
                      user_id=user_id, analysis_id=cand.get("analysis_id"))
            _concluir(t("annot.boxes_saved", n=len(caixas)))
    if st.button(t("annot.no_bee_btn"), width="stretch", key=f"{prefixo}_sem_abelha"):
        database.set_candidate_labels(cand["id"], user_id, [], "sem_abelha")
        _concluir(t("annot.flash_no_bee", id=cand["id"]), "info")
    if st.button(t("annot.discard_btn"), width="stretch", key=f"{prefixo}_descartar",
                 help=t("annot.discard_help")):
        database.set_candidate_labels(cand["id"], user_id, [], "descartado")
        _concluir(t("annot.flash_discarded", id=cand["id"]), "info")
    if sugeridas:
        st.caption(t("annot.validate_note"))
    st.caption(t("annot.auto_advance_hint"))


def _aplicar_perfil_treino():
    """on_change do perfil: leva o preset para os controles avançados."""
    p = PERFIS_TREINO.get(st.session_state.get("train_perfil"))
    if p:
        # Um modelo treinado/ativo escolhido como base (continuar o retreino)
        # é mantido; só os modelos base do Ultralytics seguem o perfil.
        if st.session_state.get("train_base") in (None, *MODELOS_BASE):
            st.session_state["train_base"] = p["base"]
        st.session_state["train_epochs"] = p["epochs"]
        st.session_state["train_imgsz"] = p["imgsz"]


def _composicao_dataset(user_id) -> str:
    """Linha com a composição das imagens rotuladas (validadas/reanotadas/negativas)."""
    b = database.candidate_breakdown(user_id)
    return t("annot.dataset_breakdown", validated=b["validado"],
             reannotated=b["reanotado"], negatives=b["sem_abelha"],
             review=b["conferencia"])


def _modelos_base_treino() -> dict:
    """{caminho/nome: rótulo} dos pontos de partida do treino.

    Os modelos base do Ultralytics (baixados na 1ª vez), os modelos já treinados
    no app (beemove_*.pt, mais recentes primeiro) e o modelo ativo, quando .pt —
    assim um retreino pode continuar a partir do modelo atual.
    """
    _, ativo = modelo_ativo()
    opcoes = {m: t("annot.base_pretrained", name=m) for m in MODELOS_BASE}
    try:
        treinados = sorted(_TRAINED_DIR.glob("beemove_*.pt"),
                           key=lambda q: q.stat().st_mtime, reverse=True)
    except Exception:
        treinados = []
    caminhos = [str(q) for q in treinados]
    if str(ativo).endswith(".pt") and os.path.exists(ativo) and str(ativo) not in caminhos:
        caminhos.insert(0, str(ativo))
    for c in caminhos:
        nome = os.path.basename(c)
        opcoes[c] = (t("annot.base_active", name=nome) if c == str(ativo)
                     else t("annot.base_trained", name=nome))
    return opcoes


def _anot_treinar(user_id, n_anotados, n_ext):
    """Treino (fine-tuning Ultralytics): perfil simples + controles avançados."""
    if not _ultralytics_disponivel():
        st.warning(t("annot.no_ultralytics"))
        return
    total_imgs = n_anotados + n_ext
    st.markdown(t("annot.train_caption", n_reannot=n_anotados, n_ext=n_ext,
                  total=total_imgs))
    st.caption(_composicao_dataset(user_id))
    disp, nome_gpu = dispositivo_treino()
    if disp == "gpu":
        st.info(t("annot.train_on_gpu", name=nome_gpu or "CUDA"))
    else:
        st.warning(t("annot.train_on_cpu"))

    perfis = {k: t(f"annot.profile_{k}") for k in PERFIS_TREINO}
    if st.session_state.get("train_perfil") not in (None, *perfis):
        st.session_state.pop("train_perfil", None)
    ult = st.session_state.get("_keep_train_perfil")
    padrao = ult if ult in perfis else ("fast" if disp == "cpu" else "balanced")
    perfil = st.segmented_control(
        t("annot.profile"), list(perfis), format_func=perfis.get,
        key="train_perfil", required=True, default=padrao,
        on_change=_aplicar_perfil_treino, help=t("annot.profile_help"))
    if perfil not in perfis:
        perfil = padrao
    st.session_state["_keep_train_perfil"] = perfil
    st.caption(t(f"annot.profile_{perfil}_desc"))

    preset = PERFIS_TREINO[perfil]
    bases = _modelos_base_treino()
    validos = {"train_base": list(bases), "train_imgsz": [320, 416, 512, 640]}
    for k, v in (("train_base", preset["base"]), ("train_epochs", preset["epochs"]),
                 ("train_imgsz", preset["imgsz"])):
        atual = st.session_state.get(k)
        if atual is None or (k in validos and atual not in validos[k]):
            st.session_state[k] = v
    with st.expander(t("annot.advanced")):
        base = st.selectbox(t("annot.base_model"), list(bases), key="train_base",
                            format_func=lambda b: bases.get(b, os.path.basename(b)),
                            help=t("annot.base_model_help"))
        cep, cimg = st.columns(2)
        epochs = cep.slider(t("annot.epochs"), 10, 300, step=10, key="train_epochs",
                            help=t("annot.epochs_help"))
        imgsz = cimg.select_slider(t("annot.imgsz"), [320, 416, 512, 640],
                                   key="train_imgsz", help=t("annot.imgsz_help"))
    st.caption(t("annot.train_summary", base=os.path.basename(base), epochs=epochs,
                 imgsz=imgsz))

    if total_imgs < 5:
        st.info(t("annot.need_more_images"))
    else:
        st.warning(t("annot.train_keep_page_open"))
        if st.button(t("annot.train_btn"), type="primary", key="train_go"):
            _anot_executar_treino(user_id, base, epochs, imgsz)
    _anot_resultado_treino(user_id)


def _anot_executar_treino(user_id, base, epochs, imgsz):
    """Roda o treino; o modelo novo fica em session_state p/ o resultado (e a ativação)."""
    import traceback

    st.session_state.pop("_anot_treino_novo", None)
    destino = erro = None
    with st.status(t("annot.training_status"), expanded=True) as caixa:
        barra = st.progress(0.0, text=t("annot.training_preparing"))
        try:
            destino, ni, nc = treinar_modelo(user_id, base, epochs, imgsz, barra)
            log_event("model_trained",
                      f"Treino Ultralytics: {destino.name} ({ni} imgs, {nc} caixas, "
                      f"base {os.path.basename(str(base))})",
                      user_id=user_id)
            caixa.update(label=t("annot.training_done_status"), state="complete",
                         expanded=False)
        except Exception as e:
            erro = e
            detalhe = "".join(traceback.format_exception(e))
            log_event("model_train_failed", f"Erro no treino: {e}",
                      user_id=user_id, level=logging.ERROR)
            caixa.update(label=t("annot.training_failed_status"), state="error",
                         expanded=False)
        finally:
            barra.empty()

    if erro is not None:
        tipo = erro_treino_amigavel(erro)
        st.error(t(f"annot.train_err_{tipo}", base=os.path.basename(str(base))))
        with st.expander(t("annot.train_error_details")):
            st.code(detalhe, language=None)
    elif destino is not None and destino.exists():
        st.session_state["_anot_treino_novo"] = str(destino)
    else:
        st.error(t("annot.train_no_file"))


def _anot_resultado_treino(user_id):
    """Resultado do último treino da sessão; admins podem ativar o modelo já aqui."""
    caminho = st.session_state.get("_anot_treino_novo")
    if not caminho or not os.path.exists(caminho):
        return
    nome = os.path.basename(caminho)
    if st.session_state.get("role") != "admin":
        st.success(t("annot.train_ok_user", name=nome))
        return
    st.success(t("annot.train_ok_admin", name=nome))
    if os.path.abspath(modelo_ativo()[1]) == os.path.abspath(caminho):
        st.caption(t("annot.trained_is_active", name=nome))
    elif st.button(t("annot.activate_trained_btn"), type="primary",
                   key="train_ativar_novo"):
        ativar_modelo(caminho, "", user_id, evento="model_activated")
        ui.flash(t("admin.model_activated", name=nome))
        st.rerun()


def _relatorio_importacao(r):
    """Resumo em linguagem simples do que veio no .zip importado."""
    if not r.get("imagens"):
        st.warning(t("annot.import_no_images", n_lbl=r.get("rotulos", 0)))
        return
    st.success(t("annot.import_report", n=r["imagens"], with_lbl=r["com_rotulo"],
                 without=r["sem_rotulo"]))
    if r["sem_rotulo"]:
        st.caption(t("annot.import_report_negatives", n=r["sem_rotulo"]))
    if r["rotulos_orfaos"]:
        st.warning(t("annot.import_report_orphans", n=r["rotulos_orfaos"]))


def _anot_dados(user_id):
    """Importar imagens externas (dataset YOLO) e exportar o dataset reanotado."""
    import zipfile

    with st.container(border=True):
        st.markdown(t("annot.import_title"))
        n_ext_i, n_ext_l = contar_externo()
        st.caption(t("annot.import_caption", n_img=n_ext_i, n_lbl=n_ext_l))
        rel = st.session_state.pop("_anot_import_rel", None)
        if isinstance(rel, dict):
            _relatorio_importacao(rel)
        n_up = st.session_state.get("_anot_zip_n", 0)
        zf = st.file_uploader(t("annot.zip_uploader"), type=["zip"],
                              key=f"ext_zip_{n_up}")
        c_imp, c_limpar = st.columns(2)
        if c_imp.button(t("annot.import_zip_btn"), key="anot_importar",
                        disabled=zf is None, type="primary", width="stretch"):
            try:
                r = importar_dataset_zip(zf.getvalue())
                log_event("dataset_imported",
                          f"Dataset externo: +{r['imagens']} imgs, +{r['rotulos']} labels",
                          user_id=user_id)
                st.session_state["_anot_import_rel"] = r
                st.session_state["_anot_zip_n"] = n_up + 1   # limpa o seletor
                st.rerun()
            except zipfile.BadZipFile:
                st.error(t("annot.import_bad_zip"))
            except Exception as e:
                st.error(t("annot.import_failed", err=e))
        if n_ext_i > 0 and c_limpar.button(t("annot.clear_external_btn"),
                                           key="anot_limpar_externo", width="stretch"):
            def _limpar():
                limpar_externo()
                log_event("dataset_cleared", "Dataset externo removido",
                          user_id=user_id)
            ui.confirm_delete(
                t("annot.clear_external_dialog_title"),
                t("annot.clear_external_dialog_body", n_img=n_ext_i, n_lbl=n_ext_l),
                _limpar, t("annot.clear_external_done"), key="dlg_limpar_externo")

    with st.container(border=True):
        st.markdown(t("annot.export_title"))
        anotados = database.list_annotated_candidates(user_id)
        res = dataset.resumo(anotados)
        st.caption(t("annot.export_caption", n_img=res["imagens"],
                     n_box=res["caixas"], n_neg=res["negativas"]))
        st.caption(_composicao_dataset(user_id))
        if res["imagens"] == 0:
            st.info(t("annot.export_empty"))
            return
        assinatura = _assinatura_dataset(anotados)
        cache = st.session_state.get("dataset_zip")
        if not isinstance(cache, dict) or cache.get("user") != user_id:
            cache = None
        desatualizado = cache is not None and cache.get("sig") != assinatura
        if desatualizado:
            st.info(t("annot.export_outdated"))
        if cache is None or desatualizado:
            rotulo = (t("annot.rebuild_zip_btn") if desatualizado
                      else t("annot.build_zip_btn"))
            if st.button(rotulo, key="anot_gerar_zip"):
                with st.spinner(t("annot.building_zip")):
                    dados = dataset.build_yolo_zip(anotados)
                cache = {"user": user_id, "sig": assinatura, "data": dados,
                         "name": f"dataset_yolo_beemove_{datetime.now():%Y%m%d}.zip"}
                st.session_state["dataset_zip"] = cache
                desatualizado = False
                log_event("dataset_exported",
                          f"Dataset YOLO ({res['imagens']} imgs, {res['caixas']} caixas)",
                          user_id=user_id)
        if cache is not None and not desatualizado:
            st.download_button(t("annot.download_zip_btn"), data=cache["data"],
                               file_name=cache["name"], mime="application/zip",
                               type="primary", key="anot_baixar_zip")
            st.caption(t("annot.download_zip_caption", name=cache["name"]))


# --- Administração: usuários e modelos ---

LIMITE_AVALIACAO = 80   # nº máx. de imagens usadas por itens_avaliacao()


def _tipo_modelo(m) -> str:
    """'original' (YOLOv4 do projeto), 'bundled', 'trained' (no app) ou 'uploaded'."""
    p = Path(m["weights"])
    try:
        no_projeto = p.resolve().parent == _MODELS_DIR.resolve()
    except Exception:
        no_projeto = False
    if no_projeto:
        return "original" if p.suffix.lower() == ".weights" else "bundled"
    return "trained" if p.name.startswith("beemove_") else "uploaded"


def _info_arquivo(caminho):
    """(data de modificação, tamanho em MB) de um arquivo; (None, None) se falhar."""
    try:
        stt = os.stat(caminho)
        return datetime.fromtimestamp(stt.st_mtime), stt.st_size / 1e6
    except OSError:
        return None, None


def render_admin(current_user_id, input_size, usar_gpu, fp16, nms, class_names):
    """Painel de administração: abas de usuários e de modelos."""
    ui.header(t("admin.title"), t("admin.subtitle"))
    aba_usuarios, aba_modelos = st.tabs([t("admin.tab_users"), t("admin.tab_models")])
    with aba_usuarios:
        _admin_usuarios(current_user_id)
    with aba_modelos:
        _admin_modelos(current_user_id, input_size, usar_gpu, fp16, nms, class_names)


def _admin_usuarios(current_user_id):
    usuarios = database.list_users()
    c_cap, c_novo = st.columns([3, 1], vertical_alignment="bottom")
    c_cap.caption(t("admin.users_caption", n=len(usuarios),
                    n_admins=database.count_admins()))
    with c_novo.popover(t("admin.new_user_btn"), width="stretch"):
        with st.form("admin_novo_user", clear_on_submit=True):
            nu = st.text_input(t("admin.username"))
            ns = st.text_input(t("admin.password"), type="password")
            papeis = {"user": t("admin.role_user"), "admin": t("admin.role_admin")}
            nr = st.selectbox(t("admin.role"), list(papeis), format_func=papeis.get)
            if st.form_submit_button(t("admin.create_user"), type="primary",
                                     width="stretch"):
                ok, msg = auth.admin_create_user(nu, ns, nr)
                if ok:
                    ui.flash(msg)
                    st.rerun()
                else:
                    st.error(msg)
    if not usuarios:
        return

    ids = [u["id"] for u in usuarios]
    df = pd.DataFrame([{
        "user": u["username"] + (t("admin.you_suffix") if u["id"] == current_user_id
                                 else ""),
        "role": (t("admin.role_admin") if u["role"] == "admin" else t("admin.role_user")),
        "analyses": u["n_analises"],
        "created": _fmt_data(u["created_at"]) if u["created_at"] else "—",
        "last_login": _fmt_data(u["last_login_at"]) if u["last_login_at"] else "—",
    } for u in usuarios])
    ult = st.session_state.get("_keep_admin_user")
    linha_padrao = ids.index(ult) if ult in ids else (
        ids.index(current_user_id) if current_user_id in ids else 0)
    evento = st.dataframe(
        df, hide_index=True, width="stretch",
        key=f"admin_users_tbl_{zlib.crc32(','.join(map(str, ids)).encode())}",
        on_select="rerun", selection_mode="single-row",
        selection_default={"selection": {"rows": [linha_padrao]}},
        column_config={
            "user": st.column_config.TextColumn(t("admin.col_user")),
            "role": st.column_config.TextColumn(t("admin.col_role")),
            "analyses": st.column_config.NumberColumn(t("admin.col_analyses")),
            "created": st.column_config.TextColumn(t("admin.col_created")),
            "last_login": st.column_config.TextColumn(t("admin.col_last_login")),
        },
        height=min(38 + 35 * len(ids), 320),
    )
    try:
        linhas = list(evento["selection"]["rows"])
    except (KeyError, TypeError):
        linhas = []
    if not linhas or not 0 <= linhas[0] < len(usuarios):
        st.info(t("admin.select_user_hint"))
        return
    alvo = usuarios[linhas[0]]
    st.session_state["_keep_admin_user"] = alvo["id"]
    eh_proprio = alvo["id"] == current_user_id

    with st.container(border=True):
        st.markdown(t("admin.selected_user_title", name=_md_esc(alvo["username"])))
        st.caption(t("admin.selected_user_caption",
                     role=(t("admin.role_admin") if alvo["role"] == "admin"
                           else t("admin.role_user")),
                     n=alvo.get("n_analises", 0) or 0,
                     last=(_fmt_data(alvo["last_login_at"])
                           if alvo.get("last_login_at") else "—")))
        novo_papel = "user" if alvo["role"] == "admin" else "admin"
        rotulo_papel = (t("admin.demote_to_user") if alvo["role"] == "admin"
                        else t("admin.promote_to_admin"))
        c1, c2 = st.columns(2)
        if c1.button(rotulo_papel, width="stretch", key="admin_toggle_role"):
            ok, msg = auth.admin_set_role(alvo["id"], novo_papel)
            if ok:
                ui.flash(msg)
                st.rerun()
            else:
                st.error(msg)
        if c2.button(t("admin.delete_user"), width="stretch", key="admin_del_user",
                     disabled=eh_proprio,
                     help=t("admin.cannot_delete_self") if eh_proprio else None):
            def _excluir_usuario(uid=alvo["id"]):
                ok, msg = auth.admin_delete_user(uid)
                return None if ok else msg
            ui.confirm_delete(
                t("admin.delete_user_dialog_title", name=alvo["username"]),
                t("admin.delete_user_dialog_body", name=ui.esc(alvo["username"]),
                  n=alvo.get("n_analises", 0) or 0),
                _excluir_usuario,
                t("admin.delete_user_done", name=alvo["username"]),
                key="dlg_excluir_usuario")

        with st.form("admin_reset_pwd", clear_on_submit=True):
            nova = st.text_input(t("admin.reset_password_label"), type="password")
            if st.form_submit_button(t("admin.reset_password_btn"), width="stretch"):
                ok, msg = auth.admin_reset_password(alvo["id"], nova)
                (st.success if ok else st.error)(msg)


def _admin_modelos(current_user_id, input_size, usar_gpu, fp16, nms, class_names):
    # --- Subir modelo pré-treinado (fora da plataforma) ---
    with st.expander(t("admin.upload_title")):
        st.caption(t("admin.upload_caption"))
        ups = st.file_uploader(t("admin.upload_files_label"),
                               type=["pt", "weights", "cfg"], accept_multiple_files=True,
                               key="admin_upload_modelo")
        if ups and st.button(t("admin.upload_btn"), key="admin_enviar_modelo"):
            try:
                salvos, erros = importar_modelo_upload(ups)
                if salvos:
                    log_event("model_uploaded", f"Upload de modelo: {', '.join(salvos)}",
                              user_id=current_user_id)
                    st.success(t("admin.upload_saved", files=" · ".join(salvos)))
                for e in erros:
                    st.warning(e)
            except Exception as e:
                st.error(t("admin.upload_failed", error=e))

    # --- Modelo de detecção ativo ---
    st.markdown(t("admin.model_title"))
    modelos = listar_modelos()
    _, weights_atual = modelo_ativo()
    if not modelos:
        st.info(t("admin.no_models_found", models_dir=_MODELS_DIR,
                  trained_dir=_TRAINED_DIR))
        return
    atual_nome = os.path.basename(weights_atual)
    cmp_res = st.session_state.get("admin_cmp")
    ultimo_f1 = {}
    if isinstance(cmp_res, dict) and isinstance(cmp_res.get("df"), pd.DataFrame):
        ultimo_f1 = {r["model"]: r["f1"] for _, r in cmp_res["df"].iterrows()
                     if pd.notna(r.get("f1"))}
    linhas_mod = []
    for m in modelos:
        data, mb = _info_arquivo(m["weights"])
        linhas_mod.append({
            "active": m["nome"] == atual_nome, "name": m["nome"],
            "type": t(f"admin.mtype_{_tipo_modelo(m)}"),
            "date": data, "size": mb, "f1": ultimo_f1.get(m["nome"]),
        })
    df_mod = pd.DataFrame(linhas_mod)
    nomes = [m["nome"] for m in modelos]
    linha_padrao = nomes.index(atual_nome) if atual_nome in nomes else 0
    colunas = ["active", "name", "type", "date", "size"]
    if ultimo_f1:
        colunas.append("f1")
    evento = st.dataframe(
        df_mod, hide_index=True, width="stretch",
        key=f"admin_models_tbl_{zlib.crc32(','.join(nomes).encode())}",
        on_select="rerun", selection_mode="single-row",
        selection_default={"selection": {"rows": [linha_padrao]}},
        column_order=colunas,
        column_config={
            "active": st.column_config.CheckboxColumn(t("admin.col_active"),
                                                      width="small"),
            "name": st.column_config.TextColumn(t("admin.col_model")),
            "type": st.column_config.TextColumn(t("admin.col_type")),
            "date": st.column_config.DatetimeColumn(
                t("admin.col_date"), format=t("history.date_column_format")),
            "size": st.column_config.NumberColumn(t("admin.col_size"),
                                                  format="%.1f MB"),
            "f1": st.column_config.NumberColumn(t("admin.col_last_f1"), format="%.2f",
                                                help=t("admin.help_last_f1")),
        },
    )
    try:
        sel = list(evento["selection"]["rows"])
    except (KeyError, TypeError):
        sel = []
    escolhido = modelos[sel[0]] if sel and 0 <= sel[0] < len(modelos) else None
    ja_ativo = escolhido is not None and escolhido["nome"] == atual_nome
    c_btn, c_cap = st.columns([1, 3], vertical_alignment="center")
    if c_btn.button(t("admin.activate_selected"), type="primary",
                    key="admin_aplicar_modelo", width="stretch",
                    disabled=escolhido is None or ja_ativo):
        ativar_modelo(escolhido["weights"], escolhido["cfg"], current_user_id)
        ui.flash(t("admin.model_activated", name=escolhido["nome"]))
        st.rerun()
    if escolhido is None:
        c_cap.caption(t("admin.select_model_hint"))
    else:
        c_cap.caption(t("admin.active_model_caption", name=_md_esc(atual_nome)))

    # --- Comparar modelos (em conjunto rotulado) ---
    st.divider()
    st.markdown(t("admin.compare_title"))
    cont = database.candidate_counts(current_user_id)
    n_rot = cont.get("anotado", 0) + cont.get("sem_abelha", 0) + contar_externo()[0]
    if n_rot == 0:
        st.info(t("admin.compare_need_labels"))
        return
    st.caption(t("admin.compare_caption", n=min(n_rot, LIMITE_AVALIACAO)))
    st.caption(t("admin.compare_train_overlap"))
    conf_aval = st.slider(t("admin.eval_confidence"), 0.1, 0.9, 0.4, 0.05,
                          key="admin_aval_conf", help=t("admin.eval_confidence_help"))
    avaliar = st.button(t("admin.evaluate_all"), key="admin_avaliar",
                        on_click=_marcar_ocupado)
    st.caption(t("admin.eval_keep_page_open"))
    if avaliar:
        with _tarefa_longa():
            id_alvo = (class_names.index(TARGET_CLASS)
                       if class_names and TARGET_CLASS in class_names else 0)
            linhas = []
            with st.status(t("admin.evaluating"), expanded=True) as caixa:
                st.write(t("admin.loading_eval_images"))
                itens = itens_avaliacao(current_user_id, limite=LIMITE_AVALIACAO)
                if not itens:
                    caixa.update(label=t("admin.compare_need_labels_short"), state="error")
                    ui.flash(t("admin.compare_need_labels_short"), "warning")
                else:
                    prog = st.progress(0.0)
                    for i, m in enumerate(modelos):
                        prog.progress(i / len(modelos),
                                      text=t("admin.evaluating_model", name=m["nome"],
                                             i=i + 1, n=len(modelos)))
                        linha = {"model": m["nome"], "type": _tipo_modelo(m),
                                 "precision": None, "recall": None, "f1": None,
                                 "iou_mean": None, "tp": None, "fp": None, "fn": None,
                                 "error": None}
                        try:
                            mdl, backend_m, erro_m = carregar_modelo(
                                input_size, usar_gpu, fp16, m["cfg"], m["weights"])
                            # .pt que não carregou cai p/ o Darknet padrão: não
                            # avaliar o modelo errado com o nome deste.
                            if mdl is None or (str(m["weights"]).endswith(".pt") and
                                               not str(backend_m).startswith("Ultralytics")):
                                raise RuntimeError(texto_erro_carga(erro_m) if erro_m
                                                   else t("admin.cmp_load_failed"))
                            met = avaliar_modelo(mdl, itens, conf_aval, nms, id_alvo)
                            linha.update({
                                "precision": round(met["precisao"], 3),
                                "recall": round(met["recall"], 3),
                                "f1": round(met["f1"], 3),
                                "iou_mean": round(met["iou_medio"], 3),
                                "tp": met["tp"], "fp": met["fp"], "fn": met["fn"],
                            })
                        except Exception as e:
                            linha["error"] = (str(e).strip().splitlines() or [repr(e)])[-1][:160]
                        linhas.append(linha)
                    prog.empty()
                    caixa.update(label=t("admin.evaluation_done", n=len(itens)),
                                 state="complete", expanded=False)
            if linhas:
                df_cmp = pd.DataFrame(linhas).sort_values(
                    "f1", ascending=False, na_position="last")
                st.session_state["admin_cmp"] = {"df": df_cmp, "n": len(itens),
                                                 "conf": conf_aval}
                log_event("models_compared", f"Comparados {len(modelos)} modelos",
                          user_id=current_user_id)

    cmp_res = st.session_state.get("admin_cmp")
    if not (isinstance(cmp_res, dict) and isinstance(cmp_res.get("df"), pd.DataFrame)):
        return
    df_cmp = cmp_res["df"].copy()
    df_cmp["type"] = df_cmp["type"].map(lambda c: t(f"admin.mtype_{c}"))
    st.caption(t("admin.cmp_result_caption", n=cmp_res["n"], conf=cmp_res["conf"]))
    colunas = ["model", "type", "f1", "precision", "recall", "tp", "fp", "fn",
               "iou_mean"]
    if df_cmp["error"].notna().any():
        colunas.append("error")
    _pct = dict(min_value=0.0, max_value=1.0, format="%.2f")
    st.dataframe(df_cmp, hide_index=True, width="stretch", column_order=colunas,
                 column_config={
                     "model": st.column_config.TextColumn(t("admin.col_model")),
                     "type": st.column_config.TextColumn(t("admin.col_type")),
                     "f1": st.column_config.ProgressColumn(
                         t("admin.col_f1"), help=t("admin.help_f1"), **_pct),
                     "precision": st.column_config.ProgressColumn(
                         t("admin.col_precision"), help=t("admin.help_precision"), **_pct),
                     "recall": st.column_config.ProgressColumn(
                         t("admin.col_recall"), help=t("admin.help_recall"), **_pct),
                     "tp": st.column_config.NumberColumn(t("admin.col_tp"),
                                                         help=t("admin.help_tp")),
                     "fp": st.column_config.NumberColumn(t("admin.col_fp"),
                                                         help=t("admin.help_fp")),
                     "fn": st.column_config.NumberColumn(t("admin.col_fn"),
                                                         help=t("admin.help_fn")),
                     "iou_mean": st.column_config.NumberColumn(
                         t("admin.col_iou_mean"), help=t("admin.help_iou"),
                         format="%.2f"),
                     "error": st.column_config.TextColumn(t("admin.col_error")),
                 })
    validos = df_cmp.dropna(subset=["f1"])
    if len(validos):
        melhor = validos.iloc[0]
        st.success(t("admin.best_f1", model=_md_esc(melhor["model"]),
                     f1=f"{melhor['f1']:.2f}"))


# --- Navegação (só a tela ativa é executada) ---

PAGINAS = ("new", "history", "review", "annotate", "admin")


def _rotulos_pagina() -> dict:
    return {
        "new": t("sidebar.tab_new_analysis"), "history": t("sidebar.tab_my_analyses"),
        "review": t("sidebar.tab_review"), "annotate": t("sidebar.tab_annotation"),
        "admin": t("sidebar.tab_admin"),
    }


# --- Trabalhos longos x navegação ---
# A navegação e o idioma são widgets do servidor: clicar neles durante um
# processamento faz o Streamlit interromper a execução (nada seria salvo).
# Botões que disparam trabalhos longos marcam a sessão como ocupada num
# on_click — que roda ANTES do script —, então a navegação e o idioma,
# desenhados antes do botão, já saem desabilitados nessa mesma execução.
_OCUPADO = "_tarefa_em_andamento"


def _marcar_ocupado():
    st.session_state[_OCUPADO] = True


def _ocupado() -> bool:
    return bool(st.session_state.get(_OCUPADO))


@contextmanager
def _tarefa_longa():
    """Envolve um trabalho longo disparado por botão com on_click=_marcar_ocupado.

    Ao terminar, libera a sessão e redesenha a página (a navegação desta
    execução foi desenhada desabilitada). Quem usa deve guardar em
    session_state tudo o que precisa continuar visível após o redesenho.
    """
    try:
        yield
    finally:
        st.session_state.pop(_OCUPADO, None)
    st.rerun()


def go_to(page: str):
    """Vai para outra tela (new/history/review/annotate/admin) e redesenha.

    O valor vai para uma chave pendente porque o seletor de navegação (widget
    "nav_page") já foi desenhado nesta execução e não pode ser alterado depois;
    render_navegacao() aplica o pedido no início da próxima execução.
    """
    if page not in PAGINAS:
        return
    st.session_state["_nav_goto"] = page
    st.query_params["page"] = page
    st.rerun()


def _ao_mudar_pagina():
    pagina = st.session_state.get("nav_page")
    if pagina is None:   # clique na opção já ativa: mantém a tela atual
        st.session_state["nav_page"] = st.session_state.get("_nav_last", "new")


def render_navegacao(is_admin: bool) -> str:
    """Navegação superior; devolve a tela ativa e a mantém em ?page= na URL."""
    paginas = [p for p in PAGINAS if p != "admin" or is_admin]
    pendente = st.session_state.pop("_nav_goto", None)
    if pendente in paginas:
        st.session_state["nav_page"] = pendente
    if st.session_state.get("nav_page") not in paginas:
        # 1º carregamento (ou link direto): ?page= da URL; senão a última tela.
        q = st.query_params.get("page")
        ultima = st.session_state.get("_nav_last")
        st.session_state["nav_page"] = (q if q in paginas else
                                        ultima if ultima in paginas else "new")
    rotulos = _rotulos_pagina()
    pagina = st.segmented_control(
        t("nav.label"), paginas, key="nav_page", required=True,
        format_func=lambda p: rotulos.get(p, p),
        label_visibility="collapsed", on_change=_ao_mudar_pagina,
        disabled=_ocupado(),   # trocar de tela interromperia o processamento
    )
    if pagina not in paginas:
        pagina = st.session_state.get("_nav_last", "new")
    st.session_state["_nav_last"] = pagina
    if st.query_params.get("page") != pagina:
        st.query_params["page"] = pagina
    return pagina


def gpu_disponivel(modelo_pt: bool) -> tuple[bool, str]:
    """(há GPU utilizável?, como foi verificado: "torch" ou "opencv").

    Modelos Darknet (.weights) rodam no OpenCV DNN, que precisa de OpenCV
    compilado com CUDA; modelos Ultralytics (.pt) rodam no PyTorch, que tem o
    próprio suporte a CUDA — por isso a verificação depende do modelo ativo.
    """
    if modelo_pt:
        try:
            import torch
            return bool(torch.cuda.is_available()), "torch"
        except Exception:
            return False, "torch"
    return cuda_disponivel(), "opencv"


def modelo_utilizavel(cfg: str, weights: str) -> bool:
    if str(weights).endswith(".pt"):
        return os.path.exists(weights)
    return os.path.exists(cfg) and os.path.exists(weights)


def _sair(user_id, username):
    log_event("logout", f"Logout de '{username}'", user_id=user_id)
    _descartar_video_atual()
    st.session_state.clear()
    st.rerun()


def render_conta(user_id, username, is_admin):
    """Linha compacta da conta: usuário + papel e um menu (senha / sair)."""
    papel = (ui.chip(t("sidebar.role_admin"), "admin") if is_admin
             else ui.chip(t("sidebar.role_user"), "info"))
    c_user, c_menu = st.columns([1, 1], vertical_alignment="center")
    c_user.markdown(f"👤 **{ui.esc(username)}**<br>{papel}", unsafe_allow_html=True)
    with c_menu.popover(t("sidebar.my_account"), icon=":material/manage_accounts:",
                        width="stretch"):
        with st.form("form_trocar_senha", clear_on_submit=True):
            st.markdown(t("sidebar.change_password_title"))
            p_atual = st.text_input(t("sidebar.current_password"), type="password")
            p_nova = st.text_input(t("sidebar.new_password"), type="password")
            p_conf = st.text_input(t("sidebar.confirm_password"), type="password")
            if st.form_submit_button(t("sidebar.change_password"), width="stretch"):
                if p_nova != p_conf:
                    st.error(t("sidebar.passwords_mismatch"))
                else:
                    ok, msg = auth.change_password(user_id, p_atual, p_nova)
                    if ok:
                        ui.flash(msg)
                        st.rerun()
                    st.error(msg)
        if st.button(t("sidebar.logout"), width="stretch", icon=":material/logout:"):
            _sair(user_id, username)


def render_hardware_status(info, tem_cuda, via, usar_gpu, backend, erro_cuda,
                           weights_ativo):
    """Explica em linguagem simples onde o processamento roda (GPU/CPU)."""
    with info:
        if not tem_cuda:
            st.caption(t("sidebar.no_gpu_plain", skip_label=t("sidebar.frame_skip")))
        elif "fallback" in backend:   # CUDA falhou no warm-up -> CPU
            st.caption(t("sidebar.gpu_failed_plain"))
        elif usar_gpu:
            st.caption(t("sidebar.gpu_ready"))
        else:
            st.caption(t("sidebar.gpu_off"))
        with st.expander(t("sidebar.tech_details"), expanded=False):
            resultado = t("sidebar.tech_yes") if tem_cuda else t("sidebar.tech_no")
            if via == "torch":
                st.caption(t("sidebar.tech_check_torch", result=resultado))
            else:
                st.caption(t("sidebar.tech_check_opencv", result=resultado))
            st.caption(t("sidebar.backend_in_use", backend=rotulo_backend(backend)))
            st.caption(t("sidebar.active_model", model=os.path.basename(weights_ativo)))
            st.caption(t("sidebar.target_class", cls=TARGET_CLASS))
            tipo_erro = getattr(erro_cuda, "tipo", None)
            if tipo_erro == "cuda":       # o backend caiu de fato para a CPU
                st.caption(t("sidebar.cpu_fallback_reason", error=erro_cuda))
            elif tipo_erro == "model":    # o .pt não carregou; usa o Darknet padrão
                st.caption(t("sidebar.model_fallback_reason",
                             error=texto_erro_carga(erro_cuda)))
            elif erro_cuda:               # nenhum modelo pôde ser carregado
                st.caption(t("sidebar.model_load_failed_reason", error=erro_cuda))
            if tipo_erro == "cuda" and "CUDA_ARCH" in erro_cuda:
                st.caption(t("sidebar.cuda_arch_mismatch"))
            elif not tem_cuda and via == "opencv":
                st.caption(t("sidebar.tech_no_cuda_hint"))


def chips_cabecalho(nome_modelo, backend, model_ok):
    """Chips do cabeçalho: arquivo do modelo ativo e onde ele está rodando."""
    if not model_ok:
        return ui.chip(t("nav.chip_no_model"), "warn")
    if "GPU" in backend:
        b = ui.chip(t("nav.chip_gpu"), "ok")
    elif "fallback" in backend:
        b = ui.chip(t("nav.chip_cpu_fallback"), "warn")
    else:
        b = ui.chip(t("nav.chip_cpu"), "warn")
    return ui.chip("🧠 " + ui.esc(nome_modelo), "info") + b


# --- Fluxo principal ---

ui.show_flash()

if "user_id" not in st.session_state:
    tela_login()
    st.stop()

user_id = st.session_state["user_id"]
username = st.session_state["username"]

ui.inject_css()
role = st.session_state.get("role", "user")
is_admin = role == "admin"

cfg_ativo, weights_ativo = modelo_ativo()
modelo_pt = str(weights_ativo).endswith(".pt")
tem_cuda, via_cuda = gpu_disponivel(modelo_pt)

with st.sidebar:
    i18n.language_selector(disabled=_ocupado())
    render_conta(user_id, username, is_admin)

    st.markdown("---")
    st.header(t("sidebar.settings"))

    with st.expander(t("sidebar.detection"), expanded=True):
        input_size = st.select_slider(
            t("sidebar.input_size"), options=[320, 416, 512], value=416,
            help=t("sidebar.input_size_help"), key="cfg_input_size",
        )
        frame_skip = st.slider(t("sidebar.frame_skip"), 1, 10, 10, key="cfg_frame_skip")
        conf = st.slider(t("sidebar.conf_threshold"), 0.1, 0.9, 0.4, 0.05,
                         key="cfg_conf")
        nms = st.slider(t("sidebar.nms_threshold"), 0.1, 0.9, 0.3, 0.05, key="cfg_nms")
        ampliar_areas = st.checkbox(
            t("sidebar.crop_detection"), value=True,
            help=t("sidebar.crop_detection_help"), key="cfg_ampliar",
        )
        gerar_video = st.checkbox(
            t("sidebar.review_video"), value=True,
            help=t("sidebar.review_video_help"), key="cfg_gerar_video",
        )
        coletar_candidatos = st.checkbox(
            t("sidebar.collect_candidates"), value=False,
            help=t("sidebar.collect_candidates_help"), key="cfg_coletar",
        )
        limiar_revisao = st.slider(
            t("sidebar.review_threshold"), 0.1, 0.9, 0.5, 0.05,
            disabled=not coletar_candidatos,
            help=t("sidebar.review_threshold_help"), key="cfg_limiar_revisao",
        )

    with st.expander(t("sidebar.tracking_areas"), expanded=True):
        zonas_def = st.session_state.get("zones") or []
        if zonas_def:
            # Com áreas definidas, cada área é uma placa: o nº é determinado por elas
            # (um único valor manda). Não faz sentido pedir o número manualmente.
            num_placas = len(zonas_def)
            st.info(t("sidebar.zones_defined_info", n=num_placas))
        else:
            num_placas = st.number_input(
                t("sidebar.num_dishes"), min_value=1, max_value=12, value=1,
                help=t("sidebar.num_dishes_help"), key="cfg_num_placas",
            )
        limiar_parada_px = st.slider(
            t("sidebar.stop_threshold"), 0, 50, 3,
            help=t("sidebar.stop_threshold_help"), key="cfg_limiar_parada",
        )

    with st.expander(t("sidebar.hardware"), expanded=False):
        # Toggles com key ignoram `value=` depois de criados. A disponibilidade
        # de GPU depende do modelo ativo (torch p/ .pt, OpenCV p/ .weights) e
        # pode mudar na sessão: nesse caso, volta ao padrão (GPU/FP16 ligados
        # quando há GPU).
        if st.session_state.get("_hw_tem_cuda") != tem_cuda:
            st.session_state["_hw_tem_cuda"] = tem_cuda
            st.session_state["cfg_usar_gpu"] = tem_cuda
            st.session_state["cfg_fp16"] = tem_cuda
        for _k in ("cfg_usar_gpu", "cfg_fp16"):
            st.session_state.setdefault(_k, tem_cuda)
        usar_gpu = st.toggle(   # valor inicial vem da sessão (acima)
            t("sidebar.use_gpu"), disabled=not tem_cuda,
            help=t("sidebar.use_gpu_help"), key="cfg_usar_gpu",
        ) and tem_cuda
        fp16 = st.toggle(
            t("sidebar.fp16"), disabled=not usar_gpu,
            help=t("sidebar.fp16_help"), key="cfg_fp16",
        ) and usar_gpu
        hw_info = st.container()

class_names = carregar_classes()
model, backend, erro_cuda = None, "—", None
if modelo_utilizavel(cfg_ativo, weights_ativo):
    try:
        model, backend, erro_cuda = carregar_modelo(input_size, usar_gpu, fp16,
                                                    cfg_ativo, weights_ativo)
    except Exception as e:   # ex.: .pt falhou e o Darknet padrão não existe
        erro_cuda = str(e).strip().splitlines()[-1] if str(e) else repr(e)
        log_event("model_load_failed", erro_cuda, level=logging.ERROR)
# Se o .pt não carregou, carregar_modelo cai para o Darknet padrão.
em_uso = (YOLO_WEIGHTS if modelo_pt and model is not None
          and not backend.startswith("Ultralytics") else weights_ativo)
render_hardware_status(hw_info, tem_cuda, via_cuda, usar_gpu, backend, erro_cuda,
                       em_uso)

ui.header(t("core.app_name"), t("core.app_subtitle"),
          aside=chips_cabecalho(os.path.basename(em_uso), backend, model is not None))
if model is None:
    st.error(t("core.no_usable_model", weights=weights_ativo,
               reason=f" ({texto_erro_carga(erro_cuda)})" if erro_cuda else ""))
elif em_uso != weights_ativo:
    st.warning(t("core.model_fallback", active=os.path.basename(weights_ativo),
                 used=os.path.basename(em_uso)))

pagina = render_navegacao(is_admin)

if pagina == "new":
    if model is None:
        st.info(t("core.new_needs_model"))
    else:
        render_nova_analise(user_id, model, backend, class_names,
                            frame_skip, input_size, conf, nms, limiar_parada_px,
                            num_placas, ampliar_areas, gerar_video,
                            coletar_candidatos, limiar_revisao,
                            nome_modelo=os.path.basename(em_uso))
elif pagina == "history":
    render_historico(user_id, username, limiar_parada_px)
elif pagina == "review":
    render_conferencia(user_id, model, class_names)
elif pagina == "annotate":
    render_anotacao(user_id)
elif pagina == "admin" and is_admin:
    render_admin(user_id, input_size, usar_gpu, fp16, nms, class_names)

# Rede de segurança: se a sessão ficou marcada como ocupada sem que o trabalho
# rodasse (ex.: a tela mudou antes do botão), reabilita a navegação.
if st.session_state.pop(_OCUPADO, None):
    st.rerun()
