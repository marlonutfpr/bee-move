"""Análises e gráficos derivados da trajetória da abelha.

Funções puras: recebem os arrays brutos (centróides + índice de frame) e o
fps do vídeo, devolvem métricas (dict) e figuras matplotlib. Como dependem
apenas dos dados brutos salvos no banco, valem tanto para uma análise recém
processada quanto para uma análise antiga reaberta — sem reprocessar o vídeo.

Quando o fps do vídeo é conhecido, as grandezas saem em segundos e px/s;
caso contrário, em frames e px/frame.
"""

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

MAX_KDE_POINTS = 2000  # subamostra os centróides p/ o KDE não dominar o plot


def series_temporais(centroides, indices, fps_video):
    """Arrays alinhados ao tempo: passo, velocidade instantânea e dist. acumulada."""
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    indices = np.asarray(indices).reshape(-1)
    if len(centroides) < 2:
        return None

    usa_segundos = bool(fps_video and fps_video > 0)
    t = ((indices - indices[0]) / fps_video if usa_segundos
         else (indices - indices[0]).astype(float))
    dx, dy = np.diff(centroides, axis=0).T
    passo = np.hypot(dx, dy)                  # (N-1,) deslocamento entre amostras
    dt = np.diff(t)
    velocidade = passo / np.where(dt > 0, dt, np.nan)
    dist_acum = np.concatenate([[0.0], np.cumsum(passo)])

    return {
        "t": t,
        "t_meio": (t[:-1] + t[1:]) / 2,       # tempo no centro de cada passo
        "passo": passo,
        "velocidade": velocidade,
        "dist_acum": dist_acum,
        "unidade_t": "s" if usa_segundos else "frames",
        "unidade_v": "px/s" if usa_segundos else "px/frame",
    }


def calcular_metricas(centroides, indices, fps_video, limiar_parada_px,
                      frame_shape=None, frames_processados=None,
                      frame_skip=None, pixels_per_mm=None) -> dict:
    """Métricas resumidas da trajetória (distância, velocidades, área, etc.).

    Se ``frames_processados`` e ``frame_skip`` forem informados, calcula também
    o tempo em que a abelha ficou detectada vs. não detectada. Se
    ``pixels_per_mm`` (> 0) for informado, acrescenta as métricas em milímetros.
    """
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    indices = np.asarray(indices).reshape(-1)
    m = {"deteccoes": len(centroides)}

    s = series_temporais(centroides, indices, fps_video)
    if s is None:
        return m

    passo, vel = s["passo"], s["velocidade"]
    distancia = float(np.nansum(passo))
    net = float(np.hypot(*(centroides[-1] - centroides[0])))
    tempo = float(s["t"][-1])

    m.update(
        unidade_t=s["unidade_t"],
        unidade_v=s["unidade_v"],
        distancia_px=distancia,
        deslocamento_liquido_px=net,
        retilineidade=net / distancia if distancia > 0 else 0.0,
        tempo_observado=tempo,
        velocidade_media=distancia / tempo if tempo > 0 else 0.0,
        velocidade_max=float(np.nanmax(vel)) if np.any(~np.isnan(vel)) else 0.0,
        velocidade_mediana=float(np.nanmedian(vel)) if np.any(~np.isnan(vel)) else 0.0,
    )

    dt = np.diff(s["t"])
    movendo = passo >= limiar_parada_px
    tempo_movendo = float(np.nansum(dt[movendo]))
    tempo_total = float(np.nansum(dt))
    m["tempo_movimento"] = tempo_movendo
    m["tempo_parado"] = max(tempo_total - tempo_movendo, 0.0)
    m["perc_movimento"] = 100 * tempo_movendo / tempo_total if tempo_total > 0 else 0.0

    # Tempo detectado vs. não detectado (frames processados sem detecção)
    usa_segundos = bool(fps_video and fps_video > 0)
    if frames_processados and frame_skip and usa_segundos:
        dt_amostra = frame_skip / fps_video
        nao_detectados = max(int(frames_processados) - len(centroides), 0)
        m["tempo_detectado"] = len(centroides) * dt_amostra
        m["tempo_nao_detectado"] = nao_detectados * dt_amostra
        total_proc = m["tempo_detectado"] + m["tempo_nao_detectado"]
        m["perc_detectado"] = 100 * m["tempo_detectado"] / total_proc if total_proc else 0.0

    if len(centroides) >= 3:
        hull = cv2.convexHull(centroides.astype(np.float32))
        m["area_explorada_px2"] = float(cv2.contourArea(hull))
    else:
        m["area_explorada_px2"] = 0.0

    if frame_shape is not None:
        area_frame = frame_shape[0] * frame_shape[1]
        m["cobertura_perc"] = (100 * m["area_explorada_px2"] / area_frame
                               if area_frame else 0.0)

    m["bbox_largura_px"] = float(np.ptp(centroides[:, 0]))
    m["bbox_altura_px"] = float(np.ptp(centroides[:, 1]))

    # Conversão para milímetros (somente com escala calibrada e tempo em segundos)
    if pixels_per_mm and pixels_per_mm > 0:
        m["pixels_per_mm"] = float(pixels_per_mm)
        m["distancia_mm"] = distancia / pixels_per_mm
        m["deslocamento_liquido_mm"] = net / pixels_per_mm
        m["bbox_largura_mm"] = m["bbox_largura_px"] / pixels_per_mm
        m["bbox_altura_mm"] = m["bbox_altura_px"] / pixels_per_mm
        m["area_explorada_mm2"] = m["area_explorada_px2"] / (pixels_per_mm ** 2)
        if usa_segundos:
            m["velocidade_media_mm_s"] = m["velocidade_media"] / pixels_per_mm
            m["velocidade_max_mm_s"] = m["velocidade_max"] / pixels_per_mm

    return m


# --- Gráficos sobre o frame do vídeo ---

def _figura_sobre_fundo(fundo):
    altura, largura = fundo.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8 * altura / largura))
    ax.imshow(cv2.cvtColor(fundo, cv2.COLOR_BGR2RGB), extent=[0, largura, altura, 0])
    ax.set_xlim(0, largura)
    ax.set_ylim(altura, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Posição X (pixels)")
    ax.set_ylabel("Posição Y (pixels)")
    return fig, ax


def plot_trajetoria_tempo(centroides, indices, fundo, fps_video):
    """Caminho colorido pelo tempo (azul→amarelo = início→fim)."""
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    if len(centroides) < 2:
        return None
    s = series_temporais(centroides, indices, fps_video)

    fig, ax = _figura_sobre_fundo(fundo)
    pts = centroides.reshape(-1, 1, 2)
    segmentos = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segmentos, cmap="viridis", linewidth=2)
    lc.set_array(s["t"][:-1])
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Tempo ({s['unidade_t']})")

    ax.plot(*centroides[0], "o", color="lime", markersize=10,
            markeredgecolor="black", label="Início")
    ax.plot(*centroides[-1], "s", color="red", markersize=10,
            markeredgecolor="black", label="Fim")
    ax.set_title("Trajetória ao longo do tempo")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_heatmap(centroides, fundo):
    """Mapa de calor (densidade de permanência) via KDE."""
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    if len(centroides) < 2:
        return None
    altura, largura = fundo.shape[:2]
    pontos = centroides
    if len(pontos) > MAX_KDE_POINTS:
        sel = np.linspace(0, len(pontos) - 1, MAX_KDE_POINTS).astype(int)
        pontos = pontos[sel]

    fig, ax = _figura_sobre_fundo(fundo)
    try:
        sns.kdeplot(x=pontos[:, 0], y=pontos[:, 1], cmap="rocket_r", fill=True,
                    thresh=0.05, alpha=0.6, bw_adjust=0.3, ax=ax)
    except Exception:
        plt.close(fig)
        return None
    ax.set_title("Mapa de calor das visitas")
    fig.tight_layout()
    return fig


def plot_area_explorada(centroides, fundo):
    """Pontos visitados + envoltória convexa (área total percorrida)."""
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    if len(centroides) < 3:
        return None
    fig, ax = _figura_sobre_fundo(fundo)
    ax.scatter(centroides[:, 0], centroides[:, 1], s=8, c="deepskyblue", alpha=0.5)
    hull = cv2.convexHull(centroides.astype(np.float32)).reshape(-1, 2)
    poligono = np.vstack([hull, hull[0]])
    ax.plot(poligono[:, 0], poligono[:, 1], "-", color="orange", linewidth=2,
            label="Envoltória convexa")
    ax.fill(poligono[:, 0], poligono[:, 1], color="orange", alpha=0.15)
    ax.set_title("Área explorada")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# --- Gráficos temporais ---

def plot_velocidade_tempo(centroides, indices, fps_video):
    s = series_temporais(centroides, indices, fps_video)
    if s is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(s["t_meio"], s["velocidade"], color="steelblue", linewidth=1, alpha=0.85)
    media = np.nanmean(s["velocidade"])
    ax.axhline(media, color="red", linestyle="--", linewidth=1,
               label=f"Média: {media:.1f} {s['unidade_v']}")
    ax.set_xlabel(f"Tempo ({s['unidade_t']})")
    ax.set_ylabel(f"Velocidade ({s['unidade_v']})")
    ax.set_title("Velocidade instantânea ao longo do tempo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_distancia_acumulada(centroides, indices, fps_video):
    s = series_temporais(centroides, indices, fps_video)
    if s is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(s["t"], s["dist_acum"], color="seagreen", linewidth=1.5)
    ax.fill_between(s["t"], s["dist_acum"], alpha=0.2, color="seagreen")
    ax.set_xlabel(f"Tempo ({s['unidade_t']})")
    ax.set_ylabel("Distância acumulada (px)")
    ax.set_title("Distância percorrida acumulada")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histograma_velocidade(centroides, indices, fps_video):
    s = series_temporais(centroides, indices, fps_video)
    if s is None:
        return None
    vel = s["velocidade"][~np.isnan(s["velocidade"])]
    if len(vel) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(vel, bins=30, color="mediumpurple", edgecolor="white")
    ax.axvline(float(np.median(vel)), color="red", linestyle="--",
               label=f"Mediana: {np.median(vel):.1f} {s['unidade_v']}")
    ax.set_xlabel(f"Velocidade ({s['unidade_v']})")
    ax.set_ylabel("Frequência (nº de passos)")
    ax.set_title("Distribuição de velocidades")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_posicao_tempo(centroides, indices, fps_video):
    s = series_temporais(centroides, indices, fps_video)
    if s is None:
        return None
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(s["t"], centroides[:, 0], label="X", color="tab:blue")
    ax.plot(s["t"], centroides[:, 1], label="Y", color="tab:orange")
    ax.set_xlabel(f"Tempo ({s['unidade_t']})")
    ax.set_ylabel("Posição (pixels)")
    ax.set_title("Posição X e Y ao longo do tempo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
