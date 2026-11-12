"""Relatório PDF consolidado das análises de um usuário.

Usa o backend PDF do matplotlib (já é dependência do projeto — sem libs novas).
Estrutura: capa → tabela-resumo → para cada análise com detecções, uma página
de métricas + os gráficos de trajetória e mapa de calor (reaproveitando
analytics). Função pura: recebe a lista de análises completas e devolve bytes.
"""

import io
import json
from datetime import datetime

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import analytics
from i18n import t

A4 = (8.27, 11.69)  # polegadas, retrato


def _rot(chave, largura=26):
    """Rótulo traduzido completado com pontos (alinhamento em monospace)."""
    return (t(chave) + " ").ljust(largura, ".") + " "


def _fundo(analise):
    if analise.get("first_frame_jpg"):
        return cv2.imdecode(
            np.frombuffer(analise["first_frame_jpg"], np.uint8), cv2.IMREAD_COLOR
        )
    return None


def _pagina_texto(pdf, titulo, linhas, subtitulo=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.94, titulo, ha="center", fontsize=18, weight="bold")
    if subtitulo:
        fig.text(0.5, 0.90, subtitulo, ha="center", fontsize=11, color="gray")
    y = 0.84
    for ln in linhas:
        fig.text(0.10, y, ln, fontsize=10.5, family="monospace", va="top")
        y -= 0.028
    fig.gca().axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _adicionar_figura(pdf, fig):
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)


def _linhas_metricas(m):
    ut, uv = m.get("unidade_t", "s"), m.get("unidade_v", "px/s")
    tem_mm = "distancia_mm" in m
    tem_vel_mm = "velocidade_media_mm_s" in m
    # Com escala calibrada, mostramos só as métricas em mm (sem duplicar px).
    if tem_mm:
        linhas = [
            t("report.scale_header", v=m['pixels_per_mm']),
            f"{_rot('report.m_detections')}{m.get('deteccoes', 0)}",
            f"{_rot('report.m_distance')}{m['distancia_mm']:.1f} mm",
            f"{_rot('report.m_net_displacement')}{m['deslocamento_liquido_mm']:.1f} mm",
            f"{_rot('report.m_straightness')}{m.get('retilineidade', 0):.2f}",
            (f"{_rot('report.m_mean_speed')}{m['velocidade_media_mm_s']:.2f} mm/s"
             if tem_vel_mm else
             f"{_rot('report.m_mean_speed')}{m.get('velocidade_media', 0):.1f} {uv}"),
            (f"{_rot('report.m_max_speed')}{m['velocidade_max_mm_s']:.2f} mm/s"
             if tem_vel_mm else
             f"{_rot('report.m_max_speed')}{m.get('velocidade_max', 0):.1f} {uv}"),
            f"{_rot('report.m_time_moving')}{m.get('perc_movimento', 0):.0f}%",
            f"{_rot('report.m_explored_area')}{m['area_explorada_mm2']:.1f} mm²",
        ]
    else:
        linhas = [
            f"{_rot('report.m_detections')}{m.get('deteccoes', 0)}",
            f"{_rot('report.m_distance')}{m.get('distancia_px', 0):.1f} px",
            f"{_rot('report.m_net_displacement')}{m.get('deslocamento_liquido_px', 0):.1f} px",
            f"{_rot('report.m_straightness')}{m.get('retilineidade', 0):.2f}",
            f"{_rot('report.m_mean_speed')}{m.get('velocidade_media', 0):.1f} {uv}",
            f"{_rot('report.m_max_speed')}{m.get('velocidade_max', 0):.1f} {uv}",
            f"{_rot('report.m_time_moving')}{m.get('perc_movimento', 0):.0f}%",
            f"{_rot('report.m_explored_area')}{m.get('area_explorada_px2', 0):.0f} px²",
        ]
    if "cobertura_perc" in m:
        linhas.append(f"{_rot('report.m_frame_coverage')}{m['cobertura_perc']:.1f}%")
    if "tempo_detectado" in m:
        linhas += [
            "",
            f"{_rot('report.m_time_detected')}{m['tempo_detectado']:.1f} {ut} "
            f"({m.get('perc_detectado', 0):.0f}%)",
            f"{_rot('report.m_time_undetected')}{m['tempo_nao_detectado']:.1f} {ut}",
            f"{_rot('report.m_time_walking')}{m.get('tempo_movimento', 0):.1f} {ut}",
            f"{_rot('report.m_time_stopped')}{m.get('tempo_parado', 0):.1f} {ut}",
        ]
    return linhas


def _parse_zones(valor):
    if not valor:
        return None
    try:
        zonas = json.loads(valor)
        return zonas if zonas else None
    except (ValueError, TypeError):
        return None


def _linhas_zonas(metricas_zonas):
    linhas = []
    for m in metricas_zonas:
        ut = m.get("unidade_t", "s")
        linhas.append(f"• {m['nome']}")
        linhas.append(f"    {_rot('report.z_detections_inside', 22)}{m['deteccoes_dentro']} "
                      f"({m['perc_deteccoes']:.1f}%)")
        if "tempo_permanencia" in m:
            linhas.append(f"    {_rot('report.z_dwell', 22)}{m['tempo_permanencia']:.1f} {ut}")
        linhas.append(f"    {_rot('report.z_visits', 22)}{m['visitas']}")
        linhas.append(f"    {_rot('report.z_distance_inside', 22)}{m['distancia_dentro_px']:.0f} px"
                      + (f" / {m['distancia_dentro_mm']:.1f} mm"
                         if "distancia_dentro_mm" in m else ""))
        linhas.append("")
    return linhas


def _pagina_resumo(pdf, analises):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.95, t("report.summary_title"), ha="center", fontsize=16, weight="bold")
    cols = ["ID", t("report.col_date"), t("report.col_video"), t("report.col_det_short"),
            "Dist.(px)", "Dist.(mm)", t("report.col_scale_short")]
    linhas = []
    for a in analises:
        escala = a.get("pixels_per_mm") or 0
        dist_mm = f"{(a.get('distance_px') or 0) / escala:.0f}" if escala > 0 else "-"
        nome = (a.get("video_name") or "")[:22]
        linhas.append([
            a.get("id"), (a.get("created_at") or "")[:16], nome,
            a.get("detections") or 0, f"{a.get('distance_px') or 0:.0f}",
            dist_mm, f"{escala:.2f}" if escala > 0 else "-",
        ])
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.85])
    ax.axis("off")
    tabela = ax.table(cellText=linhas, colLabels=cols, loc="upper center", cellLoc="center")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8)
    tabela.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(analises_completas, username, limiar_parada_px=3.0) -> bytes:
    """Gera o relatório PDF e devolve os bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Capa
        com_det = [a for a in analises_completas if (a.get("detections") or 0) > 0]
        _pagina_texto(
            pdf, t("report.cover_title"),
            [
                f"{_rot('report.cover_user')}{username}",
                f"{_rot('report.cover_generated')}"
                f"{datetime.now().strftime(t('report.datetime_fmt'))}",
                f"{_rot('report.cover_total')}{len(analises_completas)}",
                f"{_rot('report.cover_with_det')}{len(com_det)}",
            ],
            subtitulo=t("core.app_subtitle"),
        )

        if analises_completas:
            _pagina_resumo(pdf, analises_completas)

        # Páginas por análise
        for a in com_det:
            centroides = analytics_blob(a, "centroids")
            indices = analytics_blob(a, "frame_indices")
            if centroides is None or len(centroides) == 0:
                continue
            track_ids = analytics_blob(a, "track_ids")
            fundo = _fundo(a)
            zones = _parse_zones(a.get("zones"))
            fps = a.get("fps_video")
            tids = analytics.tracks_unicos(track_ids, centroides)
            titulo_analise = t("report.analysis_title", id=a.get("id"),
                               video=a.get("video_name"))
            processada = t("report.processed_at", ts=(a.get("created_at") or "")[:19])

            if len(tids) > 1:
                # Visão geral multi-placa
                linhas = [t("report.n_tracked", n=len(tids)), ""]
                for tid in tids:
                    c, idx = analytics.subset_track(centroides, indices, track_ids, tid)
                    mt = analytics.calcular_metricas(
                        c, idx, fps, limiar_parada_px,
                        frame_shape=fundo.shape if fundo is not None else None,
                        frames_processados=a.get("frames_processed"),
                        frame_skip=a.get("frame_skip"),
                        pixels_per_mm=a.get("pixels_per_mm"))
                    linhas.append(t(
                        "report.bee_line", n=tid + 1, det=mt.get("deteccoes", 0),
                        dist=mt.get("distancia_px", 0),
                        v=mt.get("velocidade_media", 0),
                        uv=mt.get("unidade_v", "px/s")))
                _pagina_texto(pdf, titulo_analise, linhas, subtitulo=processada)
                if fundo is not None:
                    _adicionar_figura(pdf, analytics.plot_trajetorias_multi(
                        centroides, indices, track_ids, fundo, fps, zones=zones))
                # Uma seção de métricas + trajetória por placa
                for tid in tids:
                    c, idx = analytics.subset_track(centroides, indices, track_ids, tid)
                    mt = analytics.calcular_metricas(
                        c, idx, fps, limiar_parada_px,
                        frame_shape=fundo.shape if fundo is not None else None,
                        frames_processados=a.get("frames_processed"),
                        frame_skip=a.get("frame_skip"),
                        pixels_per_mm=a.get("pixels_per_mm"))
                    _pagina_texto(pdf, t("report.bee_analysis_title", n=tid + 1,
                                         id=a.get("id")),
                                  _linhas_metricas(mt))
                    if fundo is not None:
                        _adicionar_figura(pdf, analytics.plot_trajetoria_tempo(
                            c, idx, fundo, fps, zones=zones))
            else:
                m = analytics.calcular_metricas(
                    centroides, indices, fps, limiar_parada_px,
                    frame_shape=fundo.shape if fundo is not None else None,
                    frames_processados=a.get("frames_processed"),
                    frame_skip=a.get("frame_skip"),
                    pixels_per_mm=a.get("pixels_per_mm"),
                )
                _pagina_texto(
                    pdf, titulo_analise,
                    _linhas_metricas(m),
                    subtitulo=processada,
                )
                if fundo is not None:
                    _adicionar_figura(
                        pdf, analytics.plot_trajetoria_tempo(
                            centroides, indices, fundo, fps, zones=zones))
                    _adicionar_figura(pdf, analytics.plot_heatmap(centroides, fundo,
                                                                  zones=zones))

            # Seção de áreas de monitoramento (presença agregada de todas as placas)
            if zones:
                mz = analytics.metricas_por_zona(
                    centroides, indices, fps, zones,
                    a.get("frame_skip"), a.get("pixels_per_mm"),
                )
                _pagina_texto(
                    pdf, t("report.zones_title", id=a.get("id")),
                    _linhas_zonas(mz),
                    subtitulo=t("report.zones_subtitle", n=len(zones)),
                )
                if fundo is not None:
                    _adicionar_figura(pdf, analytics.plot_mapa_zonas(
                        centroides, fundo, zones))
                    _adicionar_figura(pdf, analytics.plot_permanencia_zonas(mz))

        d = pdf.infodict()
        d["Title"] = t("report.pdf_title", user=username)
        d["Author"] = t("core.app_name")
    return buf.getvalue()


def analytics_blob(analise, chave):
    """Decodifica um blob numpy salvo no banco (centroids/frame_indices)."""
    import database
    return database.blob_to_ndarray(analise.get(chave))


def _linha_resumo(a):
    escala = a.get("pixels_per_mm") or 0
    dist_px = a.get("distance_px") or 0
    return {
        "ID": a.get("id"),
        t("report.col_date"): (a.get("created_at") or "")[:19],
        t("report.col_video"): a.get("video_name"),
        t("report.col_dishes"): a.get("num_targets") or 1,
        t("report.col_detections"): a.get("detections") or 0,
        t("report.col_distance_px"): round(dist_px, 1),
        t("report.col_distance_mm"): round(dist_px / escala, 1) if escala > 0 else None,
        t("report.col_mean_speed_px"): round(a.get("avg_speed_px_s") or 0, 1),
        t("report.col_scale"): round(escala, 2) if escala > 0 else None,
        t("report.col_areas"): len(_parse_zones(a.get("zones")) or []),
        "Frame skip": a.get("frame_skip"),
        t("report.col_net_input"): a.get("input_size"),
        t("report.col_confidence"): a.get("conf_threshold"),
        "NMS": a.get("nms_threshold"),
        "Backend": a.get("backend"),
        t("report.col_proc_time"): round(a.get("processing_time_s") or 0, 1),
    }


def _linhas_por_area(a, limiar_parada_px=3.0):
    """Uma linha por ÁREA (cada área = uma repetição do experimento).

    Sem áreas definidas, devolve uma única linha global. Com escala calibrada,
    as métricas saem em mm (sem as colunas em px).
    """
    centroides = analytics_blob(a, "centroids")
    indices = analytics_blob(a, "frame_indices")
    track_ids = analytics_blob(a, "track_ids")
    zones = _parse_zones(a.get("zones"))
    escala = a.get("pixels_per_mm") or 0
    fps = a.get("fps_video")
    base = {
        "ID": a.get("id"),
        t("report.col_date"): (a.get("created_at") or "")[:19],
        t("report.col_video"): a.get("video_name"),
        t("report.col_scale"): round(escala, 2) if escala > 0 else None,
        "Frame skip": a.get("frame_skip"),
        t("report.col_net_input"): a.get("input_size"),
        t("report.col_confidence"): a.get("conf_threshold"),
        "NMS": a.get("nms_threshold"),
        "Backend": a.get("backend"),
    }
    col_area, col_det = t("report.col_area"), t("report.col_detections")
    if centroides is None or len(centroides) == 0:
        return [{**base, col_area: "—", col_det: 0}]

    metr_z = (analytics.metricas_por_zona(centroides, indices, fps, zones,
                                          a.get("frame_skip"), escala or None)
              if zones else [])
    linhas = []
    for tid in analytics.tracks_unicos(track_ids, centroides):
        c, idx = analytics.subset_track(centroides, indices, track_ids, tid)
        m = analytics.calcular_metricas(
            c, idx, fps, limiar_parada_px,
            frames_processados=a.get("frames_processed"),
            frame_skip=a.get("frame_skip"), pixels_per_mm=escala or None)
        nome = (zones[tid].get("name") if zones and tid < len(zones) else None) or (
            t("report.area_default", n=tid + 1) if zones else t("report.area_global"))
        linha = {**base, col_area: nome, col_det: m.get("deteccoes", 0)}
        if escala > 0 and "distancia_mm" in m:
            linha[t("report.col_distance_mm")] = round(m["distancia_mm"], 1)
            linha[t("report.col_net_disp_mm")] = round(m.get("deslocamento_liquido_mm", 0), 1)
            if "velocidade_media_mm_s" in m:
                linha[t("report.col_mean_speed_mm")] = round(m["velocidade_media_mm_s"], 2)
                linha[t("report.col_max_speed_mm")] = round(m["velocidade_max_mm_s"], 2)
            linha[t("report.col_explored_mm2")] = round(m.get("area_explorada_mm2", 0), 1)
        else:
            linha[t("report.col_distance_px")] = round(m.get("distancia_px", 0), 1)
            linha[t("report.col_net_disp_px")] = round(m.get("deslocamento_liquido_px", 0), 1)
            linha[t("report.col_mean_speed_px")] = round(m.get("velocidade_media", 0), 1)
            linha[t("report.col_explored_px2")] = round(m.get("area_explorada_px2", 0), 0)
        linha[t("report.col_straightness")] = round(m.get("retilineidade", 0), 2)
        linha[t("report.col_time_moving_pct")] = round(m.get("perc_movimento", 0), 0)
        if "tempo_detectado" in m:
            linha[t("report.col_time_detected_s")] = round(m["tempo_detectado"], 1)
        # Presença na área (quando há zonas): permanência e visitas.
        if zones and tid < len(metr_z):
            mz = metr_z[tid]
            if "tempo_permanencia" in mz:
                linha[t("report.col_dwell_s")] = round(mz["tempo_permanencia"], 1)
            linha[t("report.col_visits")] = mz.get("visitas")
        linhas.append(linha)
    return linhas


def build_excel(analises_completas, username, limiar_parada_px=3.0) -> bytes:
    """Planilha .xlsx com UMA LINHA POR ÁREA (repetição). Duas abas:
    'Por área' (detalhada) e 'Resumo' (uma linha por análise)."""
    import pandas as pd
    from openpyxl.styles import Font, PatternFill, Alignment

    linhas_area = []
    for a in analises_completas:
        linhas_area.extend(_linhas_por_area(a, limiar_parada_px))
    df_area = pd.DataFrame(linhas_area)
    df_resumo = pd.DataFrame([_linha_resumo(a) for a in analises_completas])

    def _formatar(ws):
        cabecalho = Font(bold=True, color="FFFFFF")
        fundo = PatternFill("solid", fgColor="E8A317")
        for celula in ws[1]:
            celula.font = cabecalho
            celula.fill = fundo
            celula.alignment = Alignment(horizontal="center", vertical="center")
        for coluna in ws.columns:
            largura = max((len(str(c.value)) for c in coluna if c.value is not None),
                          default=8)
            ws.column_dimensions[coluna[0].column_letter].width = min(largura + 3, 42)
        ws.freeze_panes = "A2"

    aba_area, aba_resumo = t("report.sheet_by_area"), t("report.sheet_summary")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xls:
        (df_area if not df_area.empty
         else pd.DataFrame({t("report.no_data"): []})).to_excel(
            xls, index=False, sheet_name=aba_area)
        (df_resumo if not df_resumo.empty
         else pd.DataFrame({t("report.no_analyses"): []})).to_excel(
            xls, index=False, sheet_name=aba_resumo)
        for nome_aba in (aba_area, aba_resumo):
            _formatar(xls.sheets[nome_aba])
    return buf.getvalue()
