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

A4 = (8.27, 11.69)  # polegadas, retrato


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
            f"[Escala {m['pixels_per_mm']:.2f} px/mm — valores em mm]",
            f"Detecções ................ {m.get('deteccoes', 0)}",
            f"Distância percorrida ..... {m['distancia_mm']:.1f} mm",
            f"Deslocamento líquido ..... {m['deslocamento_liquido_mm']:.1f} mm",
            f"Retilineidade ............ {m.get('retilineidade', 0):.2f}",
            (f"Velocidade média ......... {m['velocidade_media_mm_s']:.2f} mm/s"
             if tem_vel_mm else
             f"Velocidade média ......... {m.get('velocidade_media', 0):.1f} {uv}"),
            (f"Velocidade máxima ........ {m['velocidade_max_mm_s']:.2f} mm/s"
             if tem_vel_mm else
             f"Velocidade máxima ........ {m.get('velocidade_max', 0):.1f} {uv}"),
            f"Tempo em movimento ....... {m.get('perc_movimento', 0):.0f}%",
            f"Área explorada ........... {m['area_explorada_mm2']:.1f} mm²",
        ]
    else:
        linhas = [
            f"Detecções ................ {m.get('deteccoes', 0)}",
            f"Distância percorrida ..... {m.get('distancia_px', 0):.1f} px",
            f"Deslocamento líquido ..... {m.get('deslocamento_liquido_px', 0):.1f} px",
            f"Retilineidade ............ {m.get('retilineidade', 0):.2f}",
            f"Velocidade média ......... {m.get('velocidade_media', 0):.1f} {uv}",
            f"Velocidade máxima ........ {m.get('velocidade_max', 0):.1f} {uv}",
            f"Tempo em movimento ....... {m.get('perc_movimento', 0):.0f}%",
            f"Área explorada ........... {m.get('area_explorada_px2', 0):.0f} px²",
        ]
    if "cobertura_perc" in m:
        linhas.append(f"Cobertura do quadro ...... {m['cobertura_perc']:.1f}%")
    if "tempo_detectado" in m:
        linhas += [
            "",
            f"Tempo detectado .......... {m['tempo_detectado']:.1f} {ut} "
            f"({m.get('perc_detectado', 0):.0f}%)",
            f"Tempo não detectado ...... {m['tempo_nao_detectado']:.1f} {ut}",
            f"Tempo caminhando ......... {m.get('tempo_movimento', 0):.1f} {ut}",
            f"Tempo parado ............. {m.get('tempo_parado', 0):.1f} {ut}",
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
        linhas.append(f"    Detecções dentro ..... {m['deteccoes_dentro']} "
                      f"({m['perc_deteccoes']:.1f}%)")
        if "tempo_permanencia" in m:
            linhas.append(f"    Permanência .......... {m['tempo_permanencia']:.1f} {ut}")
        linhas.append(f"    Visitas (entradas) ... {m['visitas']}")
        linhas.append(f"    Distância dentro ..... {m['distancia_dentro_px']:.0f} px"
                      + (f" / {m['distancia_dentro_mm']:.1f} mm"
                         if "distancia_dentro_mm" in m else ""))
        linhas.append("")
    return linhas


def _pagina_resumo(pdf, analises):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.95, "Resumo das análises", ha="center", fontsize=16, weight="bold")
    cols = ["ID", "Data", "Vídeo", "Det.", "Dist.(px)", "Dist.(mm)", "Esc.(px/mm)"]
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
            pdf, "Relatório de Análises — Bee Tracker",
            [
                f"Usuário .................. {username}",
                f"Gerado em ................ {datetime.now():%d/%m/%Y %H:%M}",
                f"Total de análises ........ {len(analises_completas)}",
                f"Análises com detecções ... {len(com_det)}",
            ],
            subtitulo="Detecção e rastreamento de abelha com YOLOv4",
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

            if len(tids) > 1:
                # Visão geral multi-placa
                linhas = [f"{len(tids)} placas/abelhas rastreadas", ""]
                for t in tids:
                    c, idx = analytics.subset_track(centroides, indices, track_ids, t)
                    mt = analytics.calcular_metricas(
                        c, idx, fps, limiar_parada_px,
                        frame_shape=fundo.shape if fundo is not None else None,
                        frames_processados=a.get("frames_processed"),
                        frame_skip=a.get("frame_skip"),
                        pixels_per_mm=a.get("pixels_per_mm"))
                    linhas.append(
                        f"Abelha {t + 1}: {mt.get('deteccoes', 0)} det · "
                        f"{mt.get('distancia_px', 0):.0f} px · "
                        f"{mt.get('velocidade_media', 0):.1f} {mt.get('unidade_v', 'px/s')}")
                _pagina_texto(
                    pdf, f"Análise #{a.get('id')} — {a.get('video_name')}",
                    linhas, subtitulo=f"Processada em {(a.get('created_at') or '')[:19]}")
                if fundo is not None:
                    _adicionar_figura(pdf, analytics.plot_trajetorias_multi(
                        centroides, indices, track_ids, fundo, fps, zones=zones))
                # Uma seção de métricas + trajetória por placa
                for t in tids:
                    c, idx = analytics.subset_track(centroides, indices, track_ids, t)
                    mt = analytics.calcular_metricas(
                        c, idx, fps, limiar_parada_px,
                        frame_shape=fundo.shape if fundo is not None else None,
                        frames_processados=a.get("frames_processed"),
                        frame_skip=a.get("frame_skip"),
                        pixels_per_mm=a.get("pixels_per_mm"))
                    _pagina_texto(pdf, f"Abelha {t + 1} — Análise #{a.get('id')}",
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
                    pdf, f"Análise #{a.get('id')} — {a.get('video_name')}",
                    _linhas_metricas(m),
                    subtitulo=f"Processada em {(a.get('created_at') or '')[:19]}",
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
                    pdf, f"Áreas de monitoramento — Análise #{a.get('id')}",
                    _linhas_zonas(mz),
                    subtitulo=f"{len(zones)} área(s) demarcada(s)",
                )
                if fundo is not None:
                    _adicionar_figura(pdf, analytics.plot_mapa_zonas(
                        centroides, fundo, zones))
                    _adicionar_figura(pdf, analytics.plot_permanencia_zonas(mz))

        d = pdf.infodict()
        d["Title"] = f"Relatório Bee Tracker — {username}"
        d["Author"] = "Bee Tracker"
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
        "Data": (a.get("created_at") or "")[:19],
        "Vídeo": a.get("video_name"),
        "Placas": a.get("num_targets") or 1,
        "Detecções": a.get("detections") or 0,
        "Distância (px)": round(dist_px, 1),
        "Distância (mm)": round(dist_px / escala, 1) if escala > 0 else None,
        "Vel. média (px/s)": round(a.get("avg_speed_px_s") or 0, 1),
        "Escala (px/mm)": round(escala, 2) if escala > 0 else None,
        "Áreas": len(_parse_zones(a.get("zones")) or []),
        "Frame skip": a.get("frame_skip"),
        "Rede (px)": a.get("input_size"),
        "Confiança": a.get("conf_threshold"),
        "NMS": a.get("nms_threshold"),
        "Backend": a.get("backend"),
        "Tempo proc. (s)": round(a.get("processing_time_s") or 0, 1),
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
        "Data": (a.get("created_at") or "")[:19],
        "Vídeo": a.get("video_name"),
        "Escala (px/mm)": round(escala, 2) if escala > 0 else None,
        "Frame skip": a.get("frame_skip"),
        "Rede (px)": a.get("input_size"),
        "Confiança": a.get("conf_threshold"),
        "NMS": a.get("nms_threshold"),
        "Backend": a.get("backend"),
    }
    if centroides is None or len(centroides) == 0:
        return [{**base, "Área": "—", "Detecções": 0}]

    metr_z = (analytics.metricas_por_zona(centroides, indices, fps, zones,
                                          a.get("frame_skip"), escala or None)
              if zones else [])
    linhas = []
    for t in analytics.tracks_unicos(track_ids, centroides):
        c, idx = analytics.subset_track(centroides, indices, track_ids, t)
        m = analytics.calcular_metricas(
            c, idx, fps, limiar_parada_px,
            frames_processados=a.get("frames_processed"),
            frame_skip=a.get("frame_skip"), pixels_per_mm=escala or None)
        nome = (zones[t].get("name") if zones and t < len(zones) else None) or (
            f"Área {t + 1}" if zones else "Global")
        linha = {**base, "Área": nome, "Detecções": m.get("deteccoes", 0)}
        if escala > 0 and "distancia_mm" in m:
            linha["Distância (mm)"] = round(m["distancia_mm"], 1)
            linha["Desloc. líquido (mm)"] = round(m.get("deslocamento_liquido_mm", 0), 1)
            if "velocidade_media_mm_s" in m:
                linha["Vel. média (mm/s)"] = round(m["velocidade_media_mm_s"], 2)
                linha["Vel. máxima (mm/s)"] = round(m["velocidade_max_mm_s"], 2)
            linha["Área explorada (mm²)"] = round(m.get("area_explorada_mm2", 0), 1)
        else:
            linha["Distância (px)"] = round(m.get("distancia_px", 0), 1)
            linha["Desloc. líquido (px)"] = round(m.get("deslocamento_liquido_px", 0), 1)
            linha["Vel. média (px/s)"] = round(m.get("velocidade_media", 0), 1)
            linha["Área explorada (px²)"] = round(m.get("area_explorada_px2", 0), 0)
        linha["Retilineidade"] = round(m.get("retilineidade", 0), 2)
        linha["Tempo mov. (%)"] = round(m.get("perc_movimento", 0), 0)
        if "tempo_detectado" in m:
            linha["Tempo detectado (s)"] = round(m["tempo_detectado"], 1)
        # Presença na área (quando há zonas): permanência e visitas.
        if zones and t < len(metr_z):
            mz = metr_z[t]
            if "tempo_permanencia" in mz:
                linha["Permanência (s)"] = round(mz["tempo_permanencia"], 1)
            linha["Visitas"] = mz.get("visitas")
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

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xls:
        (df_area if not df_area.empty
         else pd.DataFrame({"Sem dados": []})).to_excel(
            xls, index=False, sheet_name="Por área")
        (df_resumo if not df_resumo.empty
         else pd.DataFrame({"Sem análises": []})).to_excel(
            xls, index=False, sheet_name="Resumo")
        for nome_aba in ("Por área", "Resumo"):
            _formatar(xls.sheets[nome_aba])
    return buf.getvalue()
