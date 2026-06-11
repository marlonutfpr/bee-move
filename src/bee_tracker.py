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

import logging
import os
import queue
import shutil
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
    centroides: np.ndarray      # (N, 2) posições (x, y) da abelha
    indices_frames: np.ndarray  # frame de origem de cada centróide
    primeiro_frame: np.ndarray
    total_frames: int
    frames_processados: int
    fps_video: float
    tempo_s: float
    fps_processamento: float


def processar_video(video_path, model, class_names, frame_skip, conf, nms):
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

    total = max(leitor.total_frames, 1)
    barra = st.progress(0)
    status = st.empty()
    leitor.start()

    centroides, indices_frames = [], []
    primeiro_frame = None
    processados = 0
    inicio = time.perf_counter()

    while True:
        item = leitor.fila.get()
        if item is None:
            break
        indice, frame = item
        if primeiro_frame is None:
            primeiro_frame = frame.copy()

        classes, scores, boxes = model.detect(frame, conf, nms)

        if len(scores) > 0:
            scores = np.asarray(scores).reshape(-1)
            classes = np.asarray(classes).reshape(-1)
            boxes = np.asarray(boxes).reshape(-1, 4)
            if id_alvo >= 0:
                mascara = classes == id_alvo
                scores, boxes = scores[mascara], boxes[mascara]
            if len(scores) > 0:
                x, y, w, h = boxes[int(np.argmax(scores))]
                centroides.append((x + w / 2.0, y + h / 2.0))
                indices_frames.append(indice)

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

    return Resultado(
        centroides=np.asarray(centroides, dtype=float).reshape(-1, 2),
        indices_frames=np.asarray(indices_frames, dtype=int),
        primeiro_frame=primeiro_frame,
        total_frames=leitor.total_frames,
        frames_processados=processados,
        fps_video=leitor.fps,
        tempo_s=tempo,
        fps_processamento=processados / max(tempo, 1e-6),
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


def render_analytics(centroides, indices, fundo, fps_video, limiar_parada_px,
                     frames_processados=None, frame_skip=None, pixels_per_mm=None):
    """Cartões de métricas + abas de gráficos a partir dos dados brutos."""
    metr = analytics.calcular_metricas(
        centroides, indices, fps_video, limiar_parada_px,
        frame_shape=fundo.shape if fundo is not None else None,
        frames_processados=frames_processados, frame_skip=frame_skip,
        pixels_per_mm=pixels_per_mm,
    )
    uv = metr.get("unidade_v", "px/s")
    ut = metr.get("unidade_t", "s")

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
                  delta=f"{metr.get('perc_detectado', 0):.0f}% dos frames", delta_color="off")
        t2.metric("Tempo não detectado", f"{metr['tempo_nao_detectado']:.1f} {ut}",
                  help="Frames processados em que a abelha não foi detectada.")
        t3.metric("Tempo caminhando", f"{metr.get('tempo_movimento', 0):.1f} {ut}")
        t4.metric("Tempo parado", f"{metr.get('tempo_parado', 0):.1f} {ut}")

    if fundo is None:
        st.info("Sem o frame de fundo não é possível desenhar os gráficos espaciais.")
        return

    aba_traj, aba_vel, aba_cob = st.tabs(
        ["🗺️ Trajetória", "📈 Velocidade & Tempo", "🌡️ Cobertura espacial"]
    )

    with aba_traj:
        _mostrar(analytics.plot_trajetoria_tempo(centroides, indices, fundo, fps_video),
                 "A trajetória precisa de ao menos 2 detecções.")
        with st.expander("Posição X / Y ao longo do tempo"):
            _mostrar(analytics.plot_posicao_tempo(centroides, indices, fps_video))

    with aba_vel:
        _mostrar(analytics.plot_velocidade_tempo(centroides, indices, fps_video))
        col1, col2 = st.columns(2)
        with col1:
            _mostrar(analytics.plot_distancia_acumulada(centroides, indices, fps_video))
        with col2:
            _mostrar(analytics.plot_histograma_velocidade(centroides, indices, fps_video))

    with aba_cob:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Mapa de calor (densidade de permanência)**")
            _mostrar(analytics.plot_heatmap(centroides, fundo))
        with col2:
            st.markdown("**Área explorada (envoltória convexa)**")
            _mostrar(analytics.plot_area_explorada(centroides, fundo),
                     "A área explorada precisa de ao menos 3 detecções.")


# --- Persistência das análises ---

def salvar_analise(user_id, video_name, video_size, params, res: Resultado) -> int:
    ok, jpg = cv2.imencode(".jpg", res.primeiro_frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
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
            distance_px=distancia_total_px(res.centroides),
            avg_speed_px_s=velocidade_media_px_s(
                res.centroides, res.indices_frames, res.fps_video
            ),
            processing_time_s=res.tempo_s,
            fps_processing=res.fps_processamento,
        ),
        centroids_blob=database.ndarray_to_blob(res.centroides),
        frame_indices_blob=database.ndarray_to_blob(res.indices_frames),
        first_frame_jpg=jpg.tobytes() if ok else None,
    )
    log_event("analysis_completed",
              f"Análise #{analysis_id} de '{video_name}' concluída",
              user_id=user_id, analysis_id=analysis_id,
              deteccoes=len(res.centroides), tempo_s=round(res.tempo_s, 1),
              **params)
    return analysis_id


def csv_da_analise(centroides, indices_frames) -> str:
    return pd.DataFrame({
        "frame": indices_frames,
        "x": centroides[:, 0],
        "y": centroides[:, 1],
    }).to_csv(index=False)


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
                      pixels_per_mm: float | None = None):
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
        data=csv_da_analise(res.centroides, res.indices_frames),
        file_name="trajetoria.csv", mime="text/csv",
    )
    render_analytics(res.centroides, res.indices_frames, res.primeiro_frame,
                     res.fps_video, limiar_parada_px,
                     frames_processados=res.frames_processados,
                     frame_skip=frame_skip, pixels_per_mm=pixels_per_mm)


def exibir_analise_salva(analise: dict, user_id: int, limiar_parada_px: float):
    centroides = database.blob_to_ndarray(analise["centroids"])
    indices = database.blob_to_ndarray(analise["frame_indices"])
    fundo = None
    if analise["first_frame_jpg"]:
        fundo = cv2.imdecode(
            np.frombuffer(analise["first_frame_jpg"], np.uint8), cv2.IMREAD_COLOR
        )

    escala_salva = analise["pixels_per_mm"] if "pixels_per_mm" in analise.keys() else None
    st.markdown(f"### Análise #{analise['id']} — {analise['video_name']}")
    st.caption(
        f"Processada em {analise['created_at']} | backend: {analise['backend']} | "
        f"frame skip: {analise['frame_skip']} | rede: {analise['input_size']}px | "
        f"confiança: {analise['conf_threshold']} | NMS: {analise['nms_threshold']} | "
        f"tempo de processamento: {analise['processing_time_s']:.1f} s"
        + (f" | escala: {escala_salva:.2f} px/mm" if escala_salva else "")
    )

    if centroides is not None and len(centroides) > 0:
        st.download_button(
            "⬇️ Baixar trajetória (CSV)",
            data=csv_da_analise(centroides, indices),
            file_name=f"analise_{analise['id']}_trajetoria.csv",
            mime="text/csv",
        )
        render_analytics(centroides, indices, fundo,
                         analise["fps_video"], limiar_parada_px,
                         frames_processados=analise["frames_processed"],
                         frame_skip=analise["frame_skip"],
                         pixels_per_mm=escala_salva)
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

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
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
                                         value=10.0, step=0.1, key="comprimento_mm")
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


def render_nova_analise(user_id, model, backend, class_names,
                        frame_skip, input_size, conf, nms, limiar_parada_px):
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
    # Limita a exibição do vídeo a ~640x480 (st.video não tem parâmetro de tamanho)
    st.markdown(
        "<style>"
        "[data-testid='stVideo'] video, .stVideo video"
        "{max-width:640px;max-height:480px;height:auto;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.video(video_path)

    with st.expander("📏 Calibrar escala (pixels → mm) — opcional"):
        ferramenta_calibracao(video_path)

    if st.button("Processar Vídeo", type="primary"):
        params = dict(frame_skip=frame_skip, input_size=input_size,
                      conf=conf, nms=nms, backend=backend,
                      pixels_per_mm=st.session_state.get("pixels_per_mm", 0.0) or None)
        log_event("analysis_started", f"Iniciando análise de '{uploaded.name}'",
                  user_id=user_id, **params)
        try:
            with st.spinner(f"Processando vídeo (1 a cada {frame_skip} frames)..."):
                resultado = processar_video(
                    video_path, model, class_names, frame_skip, conf, nms
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
        st.success(f"Processamento concluído! Análise salva como **#{analysis_id}** "
                   "— disponível na aba 'Minhas Análises'.")

    resultado = st.session_state.get("resultado")
    if resultado is not None and resultado.primeiro_frame is not None:
        exibir_resultados(
            resultado, st.session_state.get("resultado_frame_skip", frame_skip),
            limiar_parada_px,
            pixels_per_mm=st.session_state.get("resultado_pixels_per_mm"),
        )


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
    usar_gpu = st.toggle(
        "Usar GPU (CUDA)", value=tem_cuda, disabled=not tem_cuda,
        help="Disponível apenas com OpenCV compilado com CUDA (imagem Docker GPU).",
    )
    fp16 = st.toggle(
        "Precisão FP16 na GPU", value=usar_gpu, disabled=not usar_gpu,
        help="~2x mais rápido em GPUs com Tensor Cores; precisão praticamente igual.",
    )
    input_size = st.select_slider(
        "Resolução de entrada da rede", options=[320, 416, 512], value=416,
        help="320 é mais rápido; 512 detecta melhor objetos pequenos.",
    )
    frame_skip = st.slider("Processar 1 a cada N frames", 1, 10, 3)
    conf = st.slider("Limiar de confiança", 0.1, 0.9, 0.4, 0.05)
    nms = st.slider("Limiar NMS", 0.1, 0.9, 0.3, 0.05)
    if not tem_cuda:
        st.caption("GPU indisponível: OpenCV sem suporte CUDA neste ambiente.")

    st.markdown("---")
    st.subheader("Análise")
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

aba_nova, aba_historico = st.tabs(["📹 Nova Análise", "📊 Minhas Análises"])
with aba_nova:
    render_nova_analise(user_id, model, backend, class_names,
                        frame_skip, input_size, conf, nms, limiar_parada_px)
with aba_historico:
    render_historico(user_id, username, limiar_parada_px)
