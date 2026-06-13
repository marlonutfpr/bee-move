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
import report
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

UI_UPDATE_EVERY = 10    # atualiza barra/status a cada N frames processados
QUEUE_SIZE = 32         # profundidade da fila leitor -> inferência

st.set_page_config(page_title="Bee Tracker YOLOv4", layout="wide")


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


@st.cache_resource(show_spinner="Carregando modelo YOLO...")
def carregar_modelo(input_size: int, usar_gpu: bool, fp16: bool):
    """Carrega a rede Darknet.

    Retorna (DetectionModel, descrição do backend, erro_cuda) — erro_cuda é
    None quando tudo deu certo, ou a mensagem da falha que causou o fallback
    GPU -> CPU (ex.: imagem compilada para outra arquitetura de GPU).
    """
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    erro_cuda = None

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

    def __init__(self, video_path: str, frame_skip: int):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(video_path)
        self.frame_skip = max(1, frame_skip)
        self.fila: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.aberto = self.cap.isOpened()

    def run(self):
        indice = 0
        try:
            while True:
                if indice % self.frame_skip != 0:
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
                    num_alvos=1, zones=None, ampliar_areas=False, gerar_video=False):
    leitor = LeitorDeFrames(video_path, frame_skip)
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

        if zone_paths is not None and ampliar_areas:
            # Detecção ampliada: recorta cada área e detecta ali (a abelha ocupa
            # muito mais pixels). Uma abelha por área; track_id = índice da área.
            for zi, (poly, path) in enumerate(zip(zone_polys, zone_paths)):
                res = _detectar_em_recorte(frame, model, conf, nms, id_alvo, poly, path)
                if res is not None:
                    cx, cy, box = res
                    centroides.append((cx, cy))
                    track_ids.append(zi)
                    indices_frames.append(indice)
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

        if gerar_video:
            if gravador is None:
                h0, w0 = frame.shape[:2]
                fps_saida = (leitor.fps or 10.0) / max(frame_skip, 1)
                gravador = _GravadorAnotado(w0, h0, fps_saida)
            _desenhar_anotacoes(frame, zone_polys, raw_boxes, aceitos)
            gravador.escrever(frame)

        processados += 1
        if processados % UI_UPDATE_EVERY == 0:
            barra.progress(min(indice + 1, total) / total)
            fps_proc = processados / max(time.perf_counter() - inicio, 1e-6)
            status.text(
                f"Frame {indice + 1}/{total} — "
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
        linha = {
            "Área": m["nome"],
            "Detecções": m["deteccoes_dentro"],
            "% das detecções": f"{m['perc_deteccoes']:.1f}%",
            "Visitas": m["visitas"],
            "Dist. dentro (px)": f"{m['distancia_dentro_px']:.0f}",
        }
        if "tempo_permanencia" in m:
            linha["Permanência (s)"] = f"{m['tempo_permanencia']:.1f}"
        if "distancia_dentro_mm" in m:
            linha["Dist. dentro (mm)"] = f"{m['distancia_dentro_mm']:.1f}"
        if "area_mm2" in m:
            linha["Área (mm²)"] = f"{m['area_mm2']:.0f}"
        linhas.append(linha)
    st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
    st.caption("Permanência = fração do tempo rastreado em que a abelha esteve "
               "dentro da área. Visitas = nº de entradas na área.")

    col1, col2 = st.columns(2)
    with col1:
        _mostrar(analytics.plot_mapa_zonas(centroides, fundo, zones))
    with col2:
        _mostrar(analytics.plot_permanencia_zonas(metr_z))


def render_analytics(centroides, indices, fundo, fps_video, limiar_parada_px,
                     frames_processados=None, frame_skip=None, pixels_per_mm=None,
                     zones=None):
    """Cartões de métricas + abas de gráficos a partir dos dados brutos."""
    metr = analytics.calcular_metricas(
        centroides, indices, fps_video, limiar_parada_px,
        frame_shape=fundo.shape if fundo is not None else None,
        frames_processados=frames_processados, frame_skip=frame_skip,
        pixels_per_mm=pixels_per_mm,
    )
    uv = metr.get("unidade_v", "px/s")
    ut = metr.get("unidade_t", "s")

    with st.expander("📊 Métricas resumidas", expanded=True):
        a, b, c, d = st.columns(4)
        a.metric("Detecções", metr["deteccoes"])
        b.metric("Distância total", f"{metr.get('distancia_px', 0):.0f} px")
        c.metric("Deslocamento líquido", f"{metr.get('deslocamento_liquido_px', 0):.0f} px")
        d.metric("Retilineidade", f"{metr.get('retilineidade', 0):.2f}",
                 help="Deslocamento líquido ÷ distância percorrida. "
                      "1 = linha reta; perto de 0 = trajeto muito sinuoso.")
        e, f, g, h = st.columns(4)
        e.metric("Velocidade média", f"{metr.get('velocidade_media', 0):.1f} {uv}")
        f.metric("Velocidade máxima", f"{metr.get('velocidade_max', 0):.1f} {uv}")
        g.metric("Tempo em movimento", f"{metr.get('perc_movimento', 0):.0f}%",
                 help=f"Fração do tempo com deslocamento ≥ {limiar_parada_px:.0f} px "
                      "entre amostras.")
        cobertura = metr.get("cobertura_perc")
        h.metric("Área explorada", f"{metr.get('area_explorada_px2', 0):.0f} px²",
                 delta=(f"{cobertura:.1f}% do quadro" if cobertura is not None else None),
                 delta_color="off")

        # Métricas em milímetros (quando a escala foi calibrada)
        if "distancia_mm" in metr:
            st.markdown(f"**Em milímetros** — escala de {metr['pixels_per_mm']:.2f} px/mm")
            i, j, k, m_ = st.columns(4)
            i.metric("Distância", f"{metr['distancia_mm']:.1f} mm")
            j.metric("Deslocamento líquido", f"{metr['deslocamento_liquido_mm']:.1f} mm")
            if "velocidade_media_mm_s" in metr:
                k.metric("Velocidade média", f"{metr['velocidade_media_mm_s']:.2f} mm/s")
                m_.metric("Velocidade máxima", f"{metr['velocidade_max_mm_s']:.2f} mm/s")
            else:
                k.metric("Área explorada", f"{metr['area_explorada_mm2']:.0f} mm²")
        else:
            st.caption("ℹ️ Calibre a escala (px → mm) na aba 'Nova Análise' para ver "
                       "as métricas em milímetros.")

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

        with abas[0]:
            _mostrar(analytics.plot_trajetoria_tempo(centroides, indices, fundo,
                                                     fps_video, zones=zones),
                     "A trajetória precisa de ao menos 2 detecções.")
            st.markdown("**Posição X / Y ao longo do tempo**")
            _mostrar(analytics.plot_posicao_tempo(centroides, indices, fps_video))

        with abas[1]:
            _mostrar(analytics.plot_velocidade_tempo(centroides, indices, fps_video))
            col1, col2 = st.columns(2)
            with col1:
                _mostrar(analytics.plot_distancia_acumulada(centroides, indices, fps_video))
            with col2:
                _mostrar(analytics.plot_histograma_velocidade(centroides, indices, fps_video))

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
        render_analytics(centroides, indices, fundo, fps_video, limiar_parada_px,
                         frames_processados=frames_processados, frame_skip=frame_skip,
                         pixels_per_mm=pixels_per_mm, zones=zones)
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
            linha = {
                "Placa": rotulos[t],
                "Detecções": m.get("deteccoes", 0),
                "Distância (px)": f"{m.get('distancia_px', 0):.0f}",
                "Vel. média": f"{m.get('velocidade_media', 0):.1f} {m.get('unidade_v', 'px/s')}",
                "Tempo mov.": f"{m.get('perc_movimento', 0):.0f}%",
            }
            if "distancia_mm" in m:
                linha["Distância (mm)"] = f"{m['distancia_mm']:.1f}"
            linhas.append(linha)
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)

    # Detalhamento completo de uma placa escolhida
    opcoes = {rotulos[t]: t for t in tids}
    rotulo = st.selectbox("Ver detalhes de:", list(opcoes.keys()),
                          key=f"{key_prefix}sel_track_detalhe")
    t = opcoes[rotulo]
    c, idx = analytics.subset_track(centroides, indices, track_ids, t)
    st.markdown(f"#### Detalhes — {rotulo}")
    render_analytics(c, idx, fundo, fps_video, limiar_parada_px,
                     frames_processados=frames_processados, frame_skip=frame_skip,
                     pixels_per_mm=pixels_per_mm, zones=zones)


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
    st.title("🐝 Bee Tracker — Acesso")
    st.write("Entre com a sua conta para ver e criar análises. "
             "Cada usuário enxerga apenas as próprias análises.")

    aba_entrar, aba_registrar = st.tabs(["Entrar", "Criar conta"])

    with aba_entrar:
        with st.form("form_login"):
            usuario = st.text_input("Usuário")
            senha = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar", type="primary"):
                user, erro = auth.login(usuario, senha)
                if user:
                    st.session_state["user_id"] = user["id"]
                    st.session_state["username"] = user["username"]
                    st.rerun()
                else:
                    st.error(erro)

    with aba_registrar:
        with st.form("form_registro"):
            usuario = st.text_input("Usuário", help="Mínimo de 3 caracteres")
            senha = st.text_input("Senha", type="password", help="Mínimo de 6 caracteres")
            confirma = st.text_input("Confirmar senha", type="password")
            if st.form_submit_button("Criar conta"):
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
        st.dataframe(df, hide_index=True, use_container_width=True)

        # --- Relatório geral consolidado ---
        st.markdown("#### Relatório geral")
        col_csv, col_pdf = st.columns(2)
        with col_csv:
            st.download_button(
                "⬇️ Resumo (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resumo_analises.csv", mime="text/csv",
                use_container_width=True,
            )
        with col_pdf:
            if st.button("📄 Gerar relatório (PDF)", use_container_width=True):
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
                         use_container_width=True)
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


def _detectar_em_recorte(frame, model, conf, nms, id_alvo, poligono, path, margem=0.15):
    """Detecta a abelha dentro do recorte (ampliado) de uma área.

    Recorta a caixa que envolve o polígono da área (com `margem`) e roda a YOLO
    só nesse recorte — que o dnn_DetectionModel reamostra para o tamanho da rede,
    fazendo a abelha ocupar muito mais pixels (ganho em vídeo de baixa resolução).
    Devolve (cx, cy, box) da melhor detecção DENTRO da área — box=(x,y,w,h) em
    coords do frame, para desenho — ou None se nada for detectado ali.
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
        if path.contains_point((cx, cy)) and (melhor is None or s > melhor[0]):
            melhor = (float(s), cx, cy,
                      (float(x0 + x), float(y0 + y), float(w), float(h)))
    return (melhor[1], melhor[2], melhor[3]) if melhor else None


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
                st.success(f"{len(circulos)} placa(s) detectada(s) — ajuste no canvas "
                           "(modo *Ajustar* move/exclui) e clique em *Salvar áreas*.")
                st.rerun()

    zonas = st.session_state.get("zones") or []
    if zonas:
        st.caption(f"**{len(zonas)} área(s) definidas** — confira a posição no preview:")
        fig = analytics.plot_mapa_zonas(np.empty((0, 2)), frame, zonas)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
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
                        num_placas=1, ampliar_areas=False, gerar_video=False):
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
    # Limita a altura do vídeo a 800px (st.video não tem parâmetro de tamanho)
    st.markdown(
        "<style>"
        "[data-testid='stVideo'] video, .stVideo video"
        "{max-height:800px;max-width:100%;width:auto;height:auto;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.video(video_path)

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


# --- Fluxo principal ---

if "user_id" not in st.session_state:
    tela_login()
    st.stop()

user_id = st.session_state["user_id"]
username = st.session_state["username"]

if not (os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS)):
    st.error(
        f"Arquivos YOLO não encontrados! Certifique-se de que '{YOLO_CFG}' e "
        f"'{YOLO_WEIGHTS}' estão no diretório da aplicação."
    )
    st.stop()

with st.sidebar:
    st.markdown(f"👤 Conectado como **{username}**")
    if st.button("Sair"):
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
    st.header("Configurações")
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
        gerar_video = st.checkbox(
            "Gerar vídeo de conferência das detecções", value=True,
            help="Cria um MP4 com as detecções desenhadas nos frames processados "
                 "(🟧 áreas · 🟨 detecções brutas do YOLO com confiança · 🟩 detecção "
                 "aceita/rastreada), exibido na aba '🎬 Conferência'. Ajuda a ajustar "
                 "os parâmetros. Aumenta um pouco o tempo de processamento.",
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
        ampliar_areas = st.checkbox(
            "Detecção ampliada por área", value=True,
            help="Só tem efeito quando há áreas definidas. Em cada frame, recorta "
                 "cada área e amplia para o tamanho da rede antes de detectar — a "
                 "abelha ocupa muito mais pixels, melhorando a detecção em vídeos de "
                 "baixa resolução. Mais lento (uma detecção por área por frame). "
                 "Requer reprocessar o vídeo.",
        )
        limiar_parada_px = st.slider(
            "Limiar de parada (px entre amostras)", 0, 50, 3,
            help="Deslocamentos menores que isto contam como 'parada' nas métricas "
                 "de tempo em movimento. Ajuste conforme a escala do vídeo. "
                 "Não exige reprocessar — recalcula na hora.",
        )

class_names = carregar_classes()
model, backend, erro_cuda = carregar_modelo(input_size, usar_gpu, fp16)
st.sidebar.markdown("---")
st.sidebar.info(f"Backend em uso: **{backend}**")
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

st.title("🐝 Detecção e Rastreamento de Abelha com YOLOv4")

aba_nova, aba_historico, aba_conf = st.tabs(
    ["📹 Nova Análise", "📊 Minhas Análises", "🎬 Conferência"])
with aba_nova:
    render_nova_analise(user_id, model, backend, class_names,
                        frame_skip, input_size, conf, nms, limiar_parada_px,
                        num_placas, ampliar_areas, gerar_video)
with aba_historico:
    render_historico(user_id, username, limiar_parada_px)
with aba_conf:
    render_conferencia()
