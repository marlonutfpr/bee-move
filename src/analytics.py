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
from matplotlib.path import Path as MplPath
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


# --- Múltiplas placas/abelhas (tracks) ---

def tracks_unicos(track_ids, centroides=None):
    """IDs de track presentes (ou [0] quando há um único alvo / sem track_ids)."""
    if track_ids is None:
        return [0]
    tids = np.asarray(track_ids).reshape(-1)
    if len(tids) == 0:
        return [0]
    return sorted(int(t) for t in np.unique(tids))


def subset_track(centroides, indices, track_ids, t):
    """Recorta os pontos de um único track (placa/abelha)."""
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    indices = np.asarray(indices).reshape(-1)
    if track_ids is None or len(np.asarray(track_ids).reshape(-1)) == 0:
        return centroides, indices
    mask = np.asarray(track_ids).reshape(-1) == t
    return centroides[mask], indices[mask]


def plot_trajetorias_multi(centroides, indices, track_ids, fundo, fps_video,
                           zones=None, rotulos=None):
    """Trajetória de cada placa/abelha numa cor diferente, sobre o frame.

    `rotulos` é um dict opcional {track_id: nome} (ex.: nomes das áreas).
    """
    fig, ax = _figura_sobre_fundo(fundo)
    cores = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, t in enumerate(tracks_unicos(track_ids)):
        c, _ = subset_track(centroides, indices, track_ids, t)
        if len(c) == 0:
            continue
        cor = cores[i % 10]
        rotulo = (rotulos or {}).get(t, f"Abelha {t + 1}")
        ax.plot(c[:, 0], c[:, 1], "-", color=cor, linewidth=1.2, alpha=0.85,
                marker="o", markersize=2, label=rotulo)
        ax.plot(*c[0], "o", color=cor, markersize=8, markeredgecolor="black")
    if zones:
        _desenhar_zonas(ax, zones)
    ax.set_title("Trajetórias por placa")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# --- Áreas de monitoramento (zonas) ---

def pontos_em_zona(centroides, poligono):
    """Máscara booleana (N,) indicando quais centróides estão dentro do polígono."""
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    poligono = np.asarray(poligono, float).reshape(-1, 2)
    if len(poligono) < 3 or len(centroides) == 0:
        return np.zeros(len(centroides), dtype=bool)
    return MplPath(poligono).contains_points(centroides)


def _area_poligono_px2(poligono):
    """Área do polígono em px² (fórmula do shoelace)."""
    p = np.asarray(poligono, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def metricas_por_zona(centroides, indices, fps_video, zones,
                      frame_skip=None, pixels_per_mm=None) -> list:
    """Métricas de presença/permanência da abelha em cada área demarcada.

    Para cada zona: nº de detecções dentro, % das detecções, nº de visitas
    (entradas), distância percorrida dentro, tempo de permanência (se houver
    fps + frame_skip) e área da zona. Inclui conversão para mm se calibrado.
    """
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    n = len(centroides)
    usa_segundos = bool(fps_video and fps_video > 0)
    dt_amostra = (frame_skip / fps_video) if (frame_skip and usa_segundos) else None
    passo = (np.hypot(*np.diff(centroides, axis=0).T) if n >= 2
             else np.zeros(0))

    resultados = []
    for i, z in enumerate(zones or []):
        nome = z.get("name") or f"Área {i + 1}"
        poly = z.get("points") or []
        dentro = pontos_em_zona(centroides, poly)
        n_dentro = int(dentro.sum())
        # visitas = transições de fora -> dentro
        anterior = np.concatenate([[False], dentro[:-1]]) if n else np.zeros(0, bool)
        visitas = int(np.sum(dentro & ~anterior)) if n else 0
        # distância percorrida cujo passo começa dentro da zona
        dist_px = float(passo[dentro[:-1]].sum()) if n >= 2 else 0.0
        area_px2 = _area_poligono_px2(poly)

        r = {
            "nome": nome,
            "deteccoes_dentro": n_dentro,
            "perc_deteccoes": 100 * n_dentro / n if n else 0.0,
            "visitas": visitas,
            "distancia_dentro_px": dist_px,
            "area_px2": area_px2,
            "unidade_t": "s" if usa_segundos else "frames",
        }
        if dt_amostra is not None:
            r["tempo_permanencia"] = n_dentro * dt_amostra
        if pixels_per_mm and pixels_per_mm > 0:
            r["distancia_dentro_mm"] = dist_px / pixels_per_mm
            r["area_mm2"] = area_px2 / (pixels_per_mm ** 2)
        resultados.append(r)
    return resultados


def _desenhar_zonas(ax, zones):
    """Sobrepõe os polígonos das zonas (com rótulo) sobre um eixo já montado."""
    cores = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, z in enumerate(zones or []):
        poly = np.asarray(z.get("points") or [], float).reshape(-1, 2)
        if len(poly) < 3:
            continue
        cor = cores[i % 10]
        fechado = np.vstack([poly, poly[0]])
        ax.plot(fechado[:, 0], fechado[:, 1], "-", color=cor, linewidth=2,
                label=z.get("name") or f"Área {i + 1}")
        ax.fill(poly[:, 0], poly[:, 1], color=cor, alpha=0.12)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        ax.text(cx, cy, z.get("name") or f"Área {i + 1}", color="white",
                fontsize=8, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=cor, ec="none", alpha=0.85))


def plot_mapa_zonas(centroides, fundo, zones):
    """Frame com as áreas demarcadas e os pontos visitados (verificação visual)."""
    if not zones:
        return None
    fig, ax = _figura_sobre_fundo(fundo)
    centroides = np.asarray(centroides, float).reshape(-1, 2)
    if len(centroides):
        ax.scatter(centroides[:, 0], centroides[:, 1], s=6, c="white",
                   alpha=0.35, zorder=1)
    _desenhar_zonas(ax, zones)
    ax.set_title("Áreas de monitoramento")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_permanencia_zonas(metricas_zonas):
    """Gráfico de barras: tempo (ou % de detecções) de permanência por área."""
    if not metricas_zonas:
        return None
    nomes = [m["nome"] for m in metricas_zonas]
    tem_tempo = all("tempo_permanencia" in m for m in metricas_zonas)
    if tem_tempo:
        valores = [m["tempo_permanencia"] for m in metricas_zonas]
        xlabel, fmt = "Permanência (s)", "{:.1f}"
    else:
        valores = [m["perc_deteccoes"] for m in metricas_zonas]
        xlabel, fmt = "Permanência (% das detecções)", "{:.1f}%"

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(nomes) + 1)))
    cores = plt.cm.tab10(np.linspace(0, 1, 10))[:len(nomes)]
    ax.barh(nomes, valores, color=cores)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title("Permanência por área")
    for i, v in enumerate(valores):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


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


def plot_trajetoria_tempo(centroides, indices, fundo, fps_video, zones=None):
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
    if zones:
        _desenhar_zonas(ax, zones)
    ax.set_title("Trajetória ao longo do tempo")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_heatmap(centroides, fundo, zones=None):
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
    if zones:
        _desenhar_zonas(ax, zones)
        ax.legend(loc="upper right", fontsize=8)
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
