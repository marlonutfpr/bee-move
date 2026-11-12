"""Teste de fumaça da interface com o AppTest do Streamlit.

Abre o app logado como admin, com uma análise (2 placas, 2 áreas) e dois
candidatos de anotação no banco, em cada idioma e em cada tela, e falha se:
  - alguma tela levantar exceção;
  - aparecer na tela uma chave de tradução crua (ex.: "sidebar.logout").
Também lista (sem falhar) textos com cara de português na versão em inglês.

Depois confere a navegação (só a tela ativa roda): ?page= na URL, tela de
admin bloqueada para conta comum, toast de flash(), tela de login nos dois
idiomas e se o vídeo/trecho/ferramenta da Nova Análise sobrevivem à troca de
tela. Por fim, o fluxo da Anotação (etapas, anterior/próximo, avanço
automático, .zip invalidado) e da Administração (abas, tabelas, comparação),
e a anotação dos frames sem detecção na Conferência (coleta com e sem áreas,
validar a predição, contagens).

Roda dentro da imagem do app (precisa de streamlit, cv2, ultralytics...):
    docker run --rm -v "$PWD/src:/app/src:ro" -v "$PWD/tests:/app/tests:ro" \
        bee-tracker:cpu python tests/smoke_app.py
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="beemove_smoke_"))
os.environ["BEE_DB_PATH"] = str(TMP / "bee.db")
os.environ["BEE_LOG_DIR"] = str(TMP / "logs")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import auth  # noqa: E402
import database  # noqa: E402

PREFIXOS = ("core|results|history|calib|analysis|model|annot|admin|sidebar|"
            "metrics|report|auth|dataset|ui|common|nav|home|review")
CHAVE_CRUA = re.compile(rf"\b(?:{PREFIXOS})\.[a-z0-9_]+\b")
PT_SUSPEITO = re.compile(r"(ção|ções|ões\b|\bnão\b|\bvocê\b|\bplaca|\bárea|\bAnálise)", re.I)

# As telas: nomes internos aceitos pela navegação (session_state["nav_page"]).
# Se a navegação ainda for por st.tabs, todas rodam juntas e só "tudo" importa.
PAGINAS = os.environ.get("SMOKE_PAGES", "new,history,review,annotate,admin").split(",")


def semear():
    if hasattr(database, "init_db"):
        database.init_db()
    ok, msg = auth.register("smoke", "Smoke-Test-2026!")
    assert ok, msg
    uid = database.get_user_by_username("smoke")["id"] if hasattr(
        database, "get_user_by_username") else 1
    rng = np.random.default_rng(0)
    n = 120
    idx = np.repeat(np.arange(0, n * 5, 5), 2)
    tids = np.tile([0, 1], n)
    base = np.where(tids[:, None] == 0, [[150, 150]], [[450, 150]])
    cent = base + np.cumsum(rng.normal(0, 3, (2 * n, 2)), axis=0) * 0.2
    frame = np.full((300, 600, 3), 200, np.uint8)
    cv2.circle(frame, (150, 150), 120, (90, 90, 90), 3)
    cv2.circle(frame, (450, 150), 120, (90, 90, 90), 3)
    ok, jpg = cv2.imencode(".jpg", frame)
    zonas = [{"name": "Placa 1", "points": [[30, 30], [270, 30], [270, 270], [30, 270]]},
             {"name": "Placa 2", "points": [[330, 30], [570, 30], [570, 270], [330, 270]]}]
    aid = database.save_analysis(
        user_id=uid, video_name="sintetico.mp4", video_size_bytes=1234,
        params=dict(frame_skip=5, input_size=416, conf=0.4, nms=0.3, backend="CPU",
                    pixels_per_mm=4.0, zones=json.dumps(zonas)),
        metrics=dict(fps_video=30.0, total_frames=n * 5, frames_processed=n,
                     detections=2 * n, distance_px=1000.0, avg_speed_px_s=12.0,
                     processing_time_s=10.0, fps_processing=12.0, num_targets=2),
        centroids_blob=database.ndarray_to_blob(cent.astype(np.float32)),
        frame_indices_blob=database.ndarray_to_blob(idx.astype(np.int64)),
        first_frame_jpg=jpg.tobytes(),
        track_ids_blob=database.ndarray_to_blob(tids.astype(np.int64)),
    )
    for i, motivo in enumerate(["baixa_confianca", "sem_deteccao"]):
        database.add_candidate(uid, aid, "sintetico.mp4", 10 * i, motivo,
                               0.3 if i == 0 else None,
                               [[100, 100, 30, 30, 0.3]] if i == 0 else [],
                               600, 300, jpg.tobytes(), area="Placa 1")
    return uid


def textos(at) -> list[str]:
    out = []
    for nome in ("markdown", "caption", "info", "warning", "error", "success",
                 "button", "checkbox", "toggle", "slider", "select_slider",
                 "selectbox", "radio", "text_input", "number_input", "subheader",
                 "header", "title", "metric", "expander", "tabs", "multiselect",
                 "date_input", "text_area", "toast", "exception",
                 "segmented_control", "status", "pills", "download_button"):
        try:
            elementos = list(getattr(at, nome))
        except Exception:
            continue
        for el in elementos:
            for attr in ("label", "value", "body", "help", "placeholder"):
                v = getattr(el, attr, None)
                if isinstance(v, str) and v:
                    out.append(v)
            for op in getattr(el, "options", []) or []:
                if isinstance(op, str):
                    out.append(op)
    return out


def novo_app(uid, lang, role="admin"):
    at = AppTest.from_file(str(ROOT / "src/bee_tracker.py"), default_timeout=180)
    if uid is not None:
        at.session_state["user_id"] = uid
        at.session_state["username"] = "smoke"
        at.session_state["role"] = role
    at.session_state["lang"] = lang
    return at


def rodar(uid, lang, pagina):
    at = novo_app(uid, lang)
    if pagina != "tudo":
        at.session_state["nav_page"] = pagina
    at.run()
    erros = [e.value for e in at.exception]
    if pagina != "tudo" and not erros and at.session_state["nav_page"] != pagina:
        erros.append(f"nav_page={at.session_state['nav_page']!r}, esperado {pagina!r}")
    vistos = textos(at)
    cruas = sorted({m for s in vistos for m in CHAVE_CRUA.findall(s)})
    pt = sorted({s[:70] for s in vistos if lang == "en" and PT_SUSPEITO.search(s)})
    return erros, cruas, pt


def video_sintetico() -> str:
    caminho = str(TMP / "sintetico.mp4")
    vw = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(50):   # 5 s
        frame = np.full((64, 64, 3), 200, np.uint8)
        cv2.circle(frame, (10 + i, 32), 4, (40, 40, 40), -1)
        vw.write(frame)
    vw.release()
    return caminho


def checar_navegacao(uid) -> list[str]:
    """Falhas de navegação/estado (lista vazia = tudo certo)."""
    from datetime import time as dtime
    falhas = []

    def sem_erros(at, nome):
        for e in at.exception:
            falhas.append(f"{nome}: exceção {str(e.value)[:300]}")
        cruas = sorted({m for s in textos(at) for m in CHAVE_CRUA.findall(s)})
        if cruas:
            falhas.append(f"{nome}: chaves cruas {cruas[:10]}")

    # link direto: ?page=history abre o histórico
    at = novo_app(uid, "pt")
    at.query_params["page"] = "history"
    at.run()
    sem_erros(at, "deep link")
    if at.session_state["nav_page"] != "history":
        falhas.append(f"deep link: nav_page={at.session_state['nav_page']!r}")

    # conta comum não chega à administração
    at = novo_app(uid, "pt", role="user")
    at.session_state["nav_page"] = "admin"
    at.run()
    sem_erros(at, "não-admin")
    if at.session_state["nav_page"] == "admin":
        falhas.append("não-admin: a tela de administração foi aberta")

    # flash() vira toast na execução seguinte
    at = novo_app(uid, "en")
    at.session_state["_flash"] = [("Smoke flash", "success")]
    at.run()
    sem_erros(at, "flash")
    if "Smoke flash" not in [getattr(x, "value", "") for x in at.toast]:
        falhas.append("flash: toast não apareceu")

    # excluir análise abre o diálogo de confirmação (sem excluir nada)
    at = novo_app(uid, "en")
    at.session_state["nav_page"] = "history"
    at.run()
    excluir = [b for b in at.button if b.label == "🗑️ Delete this analysis"]
    if not excluir:
        falhas.append("diálogo: botão de excluir análise não encontrado")
    else:
        excluir[0].click().run()
        sem_erros(at, "diálogo de exclusão")
        rotulos = [b.label for b in at.button]
        if "Delete permanently" not in rotulos or "Cancel" not in rotulos:
            falhas.append("diálogo: confirmação de exclusão não abriu")
        if len(database.list_analyses(uid)) != 1:
            falhas.append("diálogo: a análise foi excluída sem confirmação")

    # tela de login nos dois idiomas
    for lang in ("pt", "en"):
        at = novo_app(None, lang)
        at.run()
        sem_erros(at, f"login [{lang}]")

    # estado da Nova Análise sobrevive à troca de tela
    caminho = video_sintetico()
    at = novo_app(uid, "pt")
    at.session_state["nav_page"] = "new"
    at.session_state["video_path"] = caminho
    at.session_state["video_nome"] = "sintetico.mp4"
    at.session_state["video_tamanho"] = os.path.getsize(caminho)
    at.run()
    sem_erros(at, "nova análise (vídeo)")
    try:
        trecho = (dtime(0, 0, 1), dtime(0, 0, 3))
        at.slider(key="trecho_video").set_value(trecho).run()
        at.segmented_control(key="ferramenta_desenho").set_value("scale").run()
        at.segmented_control(key="nav_page").set_value("history").run()
        sem_erros(at, "troca p/ histórico")
        at.segmented_control(key="nav_page").set_value("new").run()
        sem_erros(at, "volta p/ nova análise")
        if at.session_state["video_path"] != caminho:
            falhas.append("troca de tela: vídeo enviado se perdeu")
        if tuple(at.slider(key="trecho_video").value) != trecho:
            falhas.append(f"troca de tela: trecho virou {at.slider(key='trecho_video').value}")
        if at.segmented_control(key="ferramenta_desenho").value != "scale":
            falhas.append("troca de tela: ferramenta de preparação se perdeu")
    except Exception as e:   # widget ausente etc.
        falhas.append(f"troca de tela: {type(e).__name__}: {e}")

    falhas += checar_processamento(uid, caminho)
    return falhas


def checar_processamento(uid, caminho) -> list[str]:
    """Nova Análise: estado vazio, trecho inválido, ferramentas e uma execução real."""
    from datetime import time as dtime
    falhas = []

    def sem_erros(at, nome):
        for e in at.exception:
            falhas.append(f"{nome}: exceção {str(e.value)[:300]}")
        cruas = sorted({m for s in textos(at) for m in CHAVE_CRUA.findall(s)})
        if cruas:
            falhas.append(f"{nome}: chaves cruas {cruas[:10]}")

    # estado vazio (sem vídeo): passo a passo em vez de "aguardando upload"
    at = novo_app(uid, "en")
    at.session_state["nav_page"] = "new"
    at.run()
    sem_erros(at, "nova análise (vazia)")
    if not any("Recording tips" in c.value for c in at.caption):
        falhas.append("estado vazio: dicas de gravação não apareceram")

    # detecção automática de placas: cria as áreas e reabre o canvas sem erro
    def chave_nome0(at):
        ger = at.session_state["_zonas_geracao"] if "_zonas_geracao" in at.session_state else 0
        return f"zona_nome_{ger}_0"

    placas = str(TMP / "placas.mp4")
    vw = cv2.VideoWriter(placas, cv2.VideoWriter_fourcc(*"mp4v"), 10, (600, 300))
    for _ in range(20):
        frame = np.full((300, 600, 3), 200, np.uint8)
        cv2.circle(frame, (150, 150), 110, (60, 60, 60), 4)
        cv2.circle(frame, (450, 150), 110, (60, 60, 60), 4)
        vw.write(frame)
    vw.release()
    at = novo_app(uid, "en")
    at.session_state["nav_page"] = "new"
    at.session_state["video_path"] = placas
    at.session_state["video_nome"] = "placas.mp4"
    at.session_state["video_tamanho"] = os.path.getsize(placas)
    at.run()
    try:
        at.segmented_control(key="ferramenta_desenho").set_value("zones").run()
        at.number_input(key="hough_n").set_value(2).run()
        at.slider(key="hough_rmax").set_value(45).run()
        [b for b in at.button if b.label == "🔍 Detect dishes"][0].click().run()
        sem_erros(at, "detectar placas")
        n = len(at.session_state["zones"]) if "zones" in at.session_state else 0
        if n != 2:
            falhas.append(f"detectar placas: {n} área(s) criada(s), esperado 2")
        # renomear e detectar de novo com outro prefixo: os campos de nome não
        # podem devolver os nomes antigos (text_input com key ignora value=)
        at.text_input(key=chave_nome0(at)).set_value("Controle").run()
        at.text_input(key="zonas_prefixo").set_value("Dish").run()
        [b for b in at.button if b.label == "🔍 Detect dishes"][0].click().run()
        sem_erros(at, "detectar placas de novo")
        nomes = [z["name"] for z in at.session_state["zones"]]
        if nomes != ["Dish 1", "Dish 2"]:
            falhas.append(f"detectar de novo: nomes {nomes}, esperado ['Dish 1', 'Dish 2']")
        if at.text_input(key=chave_nome0(at)).value != "Dish 1":
            falhas.append("detectar de novo: o campo do nome manteve o nome antigo")
        # as placas guardam a geometria de elipse (para voltar editáveis)
        if not all((z.get("forma") or {}).get("tipo") == "elipse"
                   for z in at.session_state["zones"]):
            falhas.append("detectar placas: áreas sem a geometria de elipse")
        # 🗑️ exclui só aquela área e mantém o nome da outra
        at.text_input(key=chave_nome0(at)).set_value("Controle").run()
        at.button(key="zona_excluir_1").click().run()
        sem_erros(at, "excluir área")
        nomes = [z["name"] for z in at.session_state["zones"]]
        if nomes != ["Controle"]:
            falhas.append(f"excluir área: nomes {nomes}, esperado ['Controle']")
        if at.text_input(key=chave_nome0(at)).value != "Controle":
            falhas.append("excluir área: o campo do nome perdeu o nome")
        # sair da ferramenta e voltar: as áreas continuam e o formato também
        at.radio(key="zona_formato").set_value("polygon").run()
        at.segmented_control(key="ferramenta_desenho").set_value("none").run()
        at.segmented_control(key="ferramenta_desenho").set_value("zones").run()
        sem_erros(at, "voltar à ferramenta de áreas")
        if len(at.session_state["zones"]) != 1:
            falhas.append("voltar à ferramenta: as áreas sumiram")
        if at.radio(key="zona_formato").value != "polygon":
            falhas.append("voltar à ferramenta: o formato da área não foi mantido")
        if any("not saved" in w.value for w in at.warning):
            falhas.append("voltar à ferramenta: aviso de alteração não salva sem mudança")
    except Exception as e:
        falhas.append(f"detectar placas: {type(e).__name__}: {e}")

    for lang in ("pt", "en"):
        at = novo_app(uid, lang)
        at.session_state["nav_page"] = "new"
        at.session_state["video_path"] = caminho
        at.session_state["video_nome"] = "sintetico.mp4"
        at.session_state["video_tamanho"] = os.path.getsize(caminho)
        at.run()
        nome = f"processar [{lang}]"
        sem_erros(at, nome)
        try:
            # ferramentas de preparação abrem sem erro
            for ferr in ("zones", "scale", "none"):
                at.segmented_control(key="ferramenta_desenho").set_value(ferr).run()
                sem_erros(at, f"{nome} ferramenta {ferr}")
            rotulo = "Process video" if lang == "en" else "Processar Vídeo"
            # trecho com fim <= início desabilita o botão
            at.slider(key="trecho_video").set_value((dtime(0, 0, 3), dtime(0, 0, 3))).run()
            botao = [b for b in at.button if b.label == rotulo]
            if not botao or not botao[0].disabled:
                falhas.append(f"{nome}: botão não ficou desabilitado com fim <= início")
            at.slider(key="trecho_video").set_value((dtime(0, 0, 0), dtime(0, 0, 5))).run()
            botao = [b for b in at.button if b.label == rotulo]
            if not botao or botao[0].disabled:
                falhas.append(f"{nome}: botão Processar ausente/desabilitado")
                continue
            botao[0].click().run()
            sem_erros(at, f"{nome} execução")
            # ao terminar, a navegação (bloqueada durante o processamento) volta
            if "_tarefa_em_andamento" in at.session_state:
                falhas.append(f"{nome}: sessão continuou marcada como ocupada")
            if getattr(at.segmented_control(key="nav_page"), "disabled", False):
                falhas.append(f"{nome}: navegação continuou desabilitada após processar")
            resumo = at.session_state["resultado_resumo"] if "resultado_resumo" in \
                at.session_state else None
            if not resumo or not resumo.get("id"):
                falhas.append(f"{nome}: cartão de resumo não foi gerado")
                continue
            if not any(f"#{resumo['id']}" in getattr(x, "value", "") for x in at.toast):
                falhas.append(f"{nome}: toast da análise salva não apareceu")
            ver = [b for b in at.button if b.key == "pos_run_history"]
            if not ver:
                falhas.append(f"{nome}: botão 'Ver resultados' ausente")
                continue
            ver[0].click().run()
            sem_erros(at, f"{nome} ver resultados")
            if at.session_state["nav_page"] != "history":
                falhas.append(f"{nome}: 'Ver resultados' não abriu o histórico")
        except Exception as e:
            falhas.append(f"{nome}: {type(e).__name__}: {e}")
    return falhas


def checar_resultados(uid) -> list[str]:
    """Minhas Análises: tabela-seletor, filtro, grupos de gráficos, placas,
    exportações e o estado vazio da conferência."""
    falhas = []

    def sem_erros(at, nome):
        for e in at.exception:
            falhas.append(f"{nome}: exceção {str(e.value)[:300]}")
        cruas = sorted({m for s in textos(at) for m in CHAVE_CRUA.findall(s)})
        if cruas:
            falhas.append(f"{nome}: chaves cruas {cruas[:10]}")

    for lang in ("pt", "en"):
        nome = f"resultados [{lang}]"
        at = novo_app(uid, lang)
        at.session_state["nav_page"] = "history"
        at.run()
        sem_erros(at, nome)
        try:
            tabelas = [d for d in at.dataframe if str(d.key or "").startswith("hist_tbl_")]
            if not tabelas:
                falhas.append(f"{nome}: tabela de análises sem seleção (hist_tbl_*)")
                continue
            sel = at.session_state[tabelas[0].key]["selection"]["rows"]
            if sel != [0]:
                falhas.append(f"{nome}: a análise mais recente não veio selecionada ({sel})")
            if [s for s in at.selectbox if s.key == "hist_analise_sel"]:
                falhas.append(f"{nome}: o selectbox antigo de análises ainda aparece")
            ids = sorted({a["id"] for a in database.list_analyses(uid)}, reverse=True)
            pref = f"hist{ids[0]}_"
            # grupos de gráficos (só o escolhido é desenhado)
            for g in ("speed", "coverage", "trajectory"):
                at.segmented_control(key=f"{pref}det_graf_grupo").set_value(g).run()
                sem_erros(at, f"{nome} gráficos {g}")
            # gráficos opcionais (pills) mantêm as chaves graf_*
            at.pills(key=f"{pref}det_graf_extra_traj").set_value([]).run()
            sem_erros(at, f"{nome} pills opcionais")
            if at.session_state["graf_posicao"] is not False:
                falhas.append(f"{nome}: pills não atualizou graf_posicao")
            # todas as métricas + escolha da placa
            at.toggle(key=f"{pref}todas_metricas").set_value(True).run()
            sem_erros(at, f"{nome} todas as métricas")
            at.pills(key=f"{pref}sel_track_detalhe").set_value(1).run()
            sem_erros(at, f"{nome} placa 2")
            if not any("Placa 2" in m.value for m in at.markdown):
                falhas.append(f"{nome}: detalhe da placa 2 não abriu")
            # filtro sem resultado
            at.text_input(key="hist_filtro").set_value("nao-existe-xyz").run()
            sem_erros(at, f"{nome} filtro vazio")
            if not at.info:
                falhas.append(f"{nome}: filtro sem resultado não avisou")
            at.text_input(key="hist_filtro").set_value("sint").run()
            sem_erros(at, f"{nome} filtro")
            # exportação: gerar Excel troca o botão pelo download na mesma coluna
            at.button(key="gerar_xlsx").click().run()
            sem_erros(at, f"{nome} gerar Excel")
            if not [b for b in at.download_button if b.key == "baixar_xlsx"]:
                falhas.append(f"{nome}: download do Excel não apareceu")
        except Exception as e:
            falhas.append(f"{nome}: {type(e).__name__}: {e}")

        # conferência sem vídeo: estado vazio com atalho para a Nova Análise
        at = novo_app(uid, lang)
        at.session_state["nav_page"] = "review"
        at.run()
        sem_erros(at, f"conferência vazia [{lang}]")
        try:
            at.button(key="conf_ir_nova").click().run()
            sem_erros(at, f"conferência -> nova [{lang}]")
            if at.session_state["nav_page"] != "new":
                falhas.append(f"conferência [{lang}]: botão não abriu a Nova Análise")
        except Exception as e:
            falhas.append(f"conferência [{lang}]: {type(e).__name__}: {e}")
    return falhas


def checar_anotacao_admin(uid) -> list[str]:
    """Anotação (etapas, anterior/próximo, avanço automático, export) e
    Administração (abas, usuário selecionado, tabela de modelos, comparação).
    Roda por último: muda o status dos candidatos no banco."""
    falhas = []

    def sem_erros(at, nome):
        for e in at.exception:
            falhas.append(f"{nome}: exceção {str(e.value)[:300]}")
        cruas = sorted({m for s in textos(at) for m in CHAVE_CRUA.findall(s)})
        if cruas:
            falhas.append(f"{nome}: chaves cruas {cruas[:10]}")

    def tem_texto(at, trecho):
        return any(trecho in s for s in textos(at))

    try:
        at = novo_app(uid, "en")
        at.session_state["nav_page"] = "annotate"
        at.run()
        sem_erros(at, "anotação")
        if at.segmented_control(key="anot_secao").value != "label":
            falhas.append("anotação: com pendentes, a etapa inicial não é Reanotar")
        if not tem_texto(at, "Candidate 1 of 2"):
            falhas.append("anotação: posição 'Candidate 1 of 2' não apareceu")
        at.button(key="anot_next").click().run()
        sem_erros(at, "anotação próximo")
        if not tem_texto(at, "Candidate 2 of 2"):
            falhas.append("anotação: botão Próximo não avançou")
        at.button(key="anot_prev").click().run()
        sem_erros(at, "anotação anterior")
        if not tem_texto(at, "Candidate 1 of 2"):
            falhas.append("anotação: botão Anterior não voltou")
        # avanço automático: com 'todos', marcar 'sem abelha' abre o próximo pendente
        at.segmented_control(key="anot_filtro").set_value("todos").run()
        sem_erros(at, "anotação filtro todos")
        atual = at.session_state["anot_sel"]
        at.button(key="anot_sem_abelha").click().run()
        sem_erros(at, "anotação sem abelha")
        cand = database.get_candidate(atual, uid)
        if not cand or cand["status"] != "sem_abelha":
            falhas.append("anotação: 'Sem abelha' não gravou o status")
        if at.session_state["anot_sel"] == atual:
            falhas.append("anotação: não avançou para o próximo pendente")
        # treino: perfis levam os presets para os controles avançados
        at.segmented_control(key="anot_secao").set_value("train").run()
        sem_erros(at, "anotação treinar")
        at.segmented_control(key="train_perfil").set_value("accurate").run()
        sem_erros(at, "anotação perfil preciso")
        if (at.session_state["train_epochs"], at.session_state["train_base"]) != (
                200, "yolo11s.pt"):
            falhas.append("anotação: perfil 'Preciso' não aplicou o preset")
        # dados: gerar o .zip, depois mudar as anotações invalida o .zip
        at.segmented_control(key="anot_secao").set_value("data").run()
        sem_erros(at, "anotação dados")
        at.button(key="anot_gerar_zip").click().run()
        sem_erros(at, "anotação gerar zip")
        cache = at.session_state["dataset_zip"] if "dataset_zip" in at.session_state else None
        if not cache or not str(cache.get("name", "")).startswith("dataset_yolo_beemove_"):
            falhas.append(f"anotação: .zip sem o nome esperado ({cache and cache.get('name')})")
        outro = [c["id"] for c in database.list_candidates(uid) if c["id"] != atual][0]
        database.set_candidate_labels(outro, uid, [[10, 10, 20, 20]], "anotado")
        at.run()
        sem_erros(at, "anotação zip desatualizado")
        if not [b for b in at.button if b.key == "anot_gerar_zip"] or \
                [b for b in at.download_button if b.key == "anot_baixar_zip"]:
            falhas.append("anotação: .zip antigo não foi invalidado após nova anotação")
    except Exception as e:
        falhas.append(f"anotação: {type(e).__name__}: {e}")

    try:
        at = novo_app(uid, "en")
        at.session_state["nav_page"] = "admin"
        at.run()
        sem_erros(at, "admin")
        rotulos = [tab.label for tab in at.tabs]
        if rotulos[:2] != ["👥 Users", "🧠 Models"]:
            falhas.append(f"admin: abas inesperadas {rotulos}")
        if not any(str(d.key or "").startswith("admin_users_tbl_") for d in at.dataframe):
            falhas.append("admin: tabela de usuários sem seleção")
        if not any("👤 smoke" in m.value for m in at.markdown):
            falhas.append("admin: ações do usuário selecionado não apareceram")
        if not any(str(d.key or "").startswith("admin_models_tbl_") for d in at.dataframe):
            falhas.append("admin: tabela de modelos ausente")
        ativar = [b for b in at.button if b.key == "admin_aplicar_modelo"]
        if not ativar or not ativar[0].disabled:
            falhas.append("admin: 'Ativar' deveria vir desabilitado com o modelo ativo")
        at.button(key="admin_avaliar").click().run()
        sem_erros(at, "admin avaliar")
        res = at.session_state["admin_cmp"] if "admin_cmp" in at.session_state else None
        if not res or len(res["df"]) == 0:
            falhas.append("admin: comparação não gerou resultados")
        elif res["df"]["error"].notna().all():
            falhas.append(f"admin: todos os modelos falharam: {res['df']['error'].tolist()}")
    except Exception as e:
        falhas.append(f"admin: {type(e).__name__}: {e}")
    return falhas


def _processar_e_conferir(at, nome, falhas):
    """Processa o vídeo já posto na sessão e abre a Conferência; devolve o conf_ctx."""
    botao = [b for b in at.button if b.label == "Processar Vídeo"]
    if not botao or botao[0].disabled:
        falhas.append(f"{nome}: botão Processar ausente/desabilitado")
        return None
    botao[0].click().run()
    ctx = at.session_state["conf_ctx"] if "conf_ctx" in at.session_state else None
    if not ctx or not ctx.get("frames_lidos"):
        falhas.append(f"{nome}: o processamento não guardou o contexto da conferência")
        return None
    at.segmented_control(key="nav_page").set_value("review").run()
    return ctx


def checar_conferencia() -> list[str]:
    """Conferência: coletar os frames sem detecção (frame inteiro e recorte de
    área), validar uma predição, contagens, idiomas e o atalho para o treino.
    Usa um usuário próprio para não mexer nos candidatos das outras checagens."""
    falhas = []

    def sem_erros(at, nome):
        for e in at.exception:
            falhas.append(f"{nome}: exceção {str(e.value)[:300]}")
        cruas = sorted({m for s in textos(at) for m in CHAVE_CRUA.findall(s)})
        if cruas:
            falhas.append(f"{nome}: chaves cruas {cruas[:10]}")

    ok, msg = auth.register("smoke-conf", "Smoke-Test-2026!")
    if not ok:
        return [f"conferência: não criou o usuário ({msg})"]
    uid = database.get_user_by_username("smoke-conf")["id"]
    caminho = str(TMP / "conferencia.mp4")
    vw = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(50):   # 5 s; a "abelha" some na 2ª metade (frames sem detecção)
        frame = np.full((64, 64, 3), 200, np.uint8)
        if i < 25:
            cv2.circle(frame, (10 + i, 32), 4, (40, 40, 40), -1)
        vw.write(frame)
    vw.release()
    zona = {"name": "Placa A", "points": [[4, 4], [40, 4], [40, 40], [4, 40]]}
    at = None
    for zonas in ([], [zona]):
        nome = "conferência " + ("com áreas" if zonas else "sem áreas")
        try:
            at = novo_app(uid, "pt")
            at.session_state["nav_page"] = "new"
            at.session_state["video_path"] = caminho
            at.session_state["video_nome"] = "conferencia.mp4"
            at.session_state["video_tamanho"] = os.path.getsize(caminho)
            if zonas:
                at.session_state["zones"] = zonas
            at.run()
            ctx = _processar_e_conferir(at, nome, falhas)
            sem_erros(at, nome)
            if ctx is None:
                continue
            aid = ctx["analysis_id"]
            if zonas:
                # renomear a área depois de processar não muda as chaves da coleta
                at.session_state["zones"][0]["name"] = "Renomeada"
                if [z["name"] for z in at.session_state["conf_ctx"]["zones"]] != ["Placa A"]:
                    falhas.append(f"{nome}: o contexto seguiu a área renomeada")
            if not any("Anotar frames sem detecção" in m.value for m in at.markdown):
                falhas.append(f"{nome}: seção de anotação não apareceu")
            if not [b for b in at.button if b.key == "conf_col_coletar"]:
                falhas.append(f"{nome}: botão de coleta ausente "
                              f"({[x.value[:80] for x in [*at.success, *at.info]]})")
                continue
            at.slider(key="conf_col_max").set_value(10).run()
            at.button(key="conf_col_coletar").click().run()
            sem_erros(at, f"{nome} coletar")
            if "_tarefa_em_andamento" in at.session_state:
                falhas.append(f"{nome}: sessão continuou ocupada após a coleta")
            cands = database.list_candidates(uid, analysis_id=aid)
            if not cands or len(cands) > 10:
                falhas.append(f"{nome}: {len(cands)} candidato(s) guardado(s), esperado 1..10")
                continue
            if {c["origin"] for c in cands} != {"conferencia"}:
                falhas.append(f"{nome}: origem inesperada {[c['origin'] for c in cands]}")
            tamanhos = {(c["width"], c["height"]) for c in cands}
            areas = {c["area"] for c in cands}
            if zonas and (areas != {"Placa A"}
                          or any(w >= 64 or h >= 64 for w, h in tamanhos)):
                falhas.append(f"{nome}: esperado recorte da área, veio {tamanhos} {areas}")
            if not zonas and (tamanhos != {(64, 64)} or areas != {None}):
                falhas.append(f"{nome}: esperado frame inteiro, veio {tamanhos} {areas}")
            chaves = [(c["frame_index"], c["area"]) for c in cands]
            if len(set(chaves)) != len(chaves):
                falhas.append(f"{nome}: frames/áreas repetidos na coleta")
            # 2ª coleta não repete o que já foi guardado
            if [b for b in at.button if b.key == "conf_col_coletar"]:
                at.button(key="conf_col_coletar").click().run()
                sem_erros(at, f"{nome} coletar de novo")
                novos = database.list_candidates(uid, analysis_id=aid)
                chaves = [(c["frame_index"], c["area"]) for c in novos]
                if len(set(chaves)) != len(chaves):
                    falhas.append(f"{nome}: a 2ª coleta repetiu frames/áreas")
        except Exception as e:
            falhas.append(f"{nome}: {type(e).__name__}: {e}")

    if at is None or "conf_ctx" not in at.session_state:
        return falhas
    nome = "conferência validar"
    try:
        aid = at.session_state["conf_ctx"]["analysis_id"]
        cands = database.list_candidates(uid, analysis_id=aid)
        # O menor id: o anotador abre no maior (e o canvas guarda o estado do
        # que já foi exibido), então o alvo ainda não foi desenhado.
        alvo = min(c["id"] for c in cands)
        sug = [[2.0, 3.0, 10.0, 12.0, 0.12]]
        with database.get_conn() as conn:
            conn.execute("UPDATE annotation_candidates SET pred_boxes = ?, pred_conf = ?, "
                         "reason = 'baixa_confianca' WHERE id = ?",
                         (json.dumps(sug), 0.12, alvo))
        at.run()
        sem_erros(at, nome)
        if at.session_state["rev_anot_sel"] != alvo:
            at.selectbox(key="rev_anot_sel").set_value(alvo).run()
        validar = at.button(key="rev_anot_validar")
        if validar.disabled:
            falhas.append(f"{nome}: 'Validar' desabilitado com caixas sugeridas")
        validar.click().run()
        sem_erros(at, nome)
        c = database.get_candidate(alvo, uid)
        if (c["status"], c["label_source"]) != ("anotado", "validado") or \
                json.loads(c["labels"]) != [b[:4] for b in sug]:
            falhas.append(f"{nome}: gravou {c['status']}/{c['label_source']}/{c['labels']}")
        b = database.candidate_breakdown(uid, analysis_id=aid)
        if (b["validado"], b["pendente"], b["conferencia"]) != (1, len(cands) - 1, 1):
            falhas.append(f"{nome}: contagens inesperadas {b}")
        if len(cands) > 1:
            if at.session_state["rev_anot_sel"] == alvo:
                falhas.append(f"{nome}: não avançou para o próximo pendente")
            atual = database.get_candidate(at.session_state["rev_anot_sel"], uid)
            if atual and not json.loads(atual["pred_boxes"] or "[]") and \
                    not at.button(key="rev_anot_validar").disabled:
                falhas.append(f"{nome}: 'Validar' habilitado sem caixa sugerida")
        if not any("✅ **1** validado(s)" in m.value for m in at.markdown):
            falhas.append(f"{nome}: linha de contagens não mostrou o validado")
        # já reanotado com caixas diferentes das sugeridas: Validar fica desabilitado
        # um do meio da lista: nunca exibido (canvas sem estado anterior)
        outros = sorted(c["id"] for c in cands)[1:-1]
        if outros:
            with database.get_conn() as conn:
                conn.execute("UPDATE annotation_candidates SET pred_boxes = ? WHERE id = ?",
                             (json.dumps(sug), outros[0]))
            database.set_candidate_labels(outros[0], uid, [[20, 20, 6, 6]], "anotado",
                                          source="reanotado")
            at.segmented_control(key="rev_anot_filtro").set_value("todos").run()
            at.selectbox(key="rev_anot_sel").set_value(outros[0]).run()
            sem_erros(at, f"{nome} reanotado")
            if not at.button(key="rev_anot_validar").disabled:
                falhas.append(f"{nome}: 'Validar' habilitado sobre rótulos manuais")
            at.segmented_control(key="rev_anot_filtro").set_value("pendente").run()
        # os dois idiomas, sem chaves cruas
        for lang in ("en", "pt", "en"):
            at.segmented_control(key="lang_selector").set_value(lang).run()
            sem_erros(at, f"conferência [{lang}]")
        if not any("Annotate frames without detection" in m.value for m in at.markdown):
            falhas.append("conferência [en]: título da seção não traduzido")
        # atalho para o treino, com a composição do dataset
        at.button(key="conf_ir_treinar").click().run()
        sem_erros(at, "conferência -> treino")
        if at.session_state["nav_page"] != "annotate" or \
                at.segmented_control(key="anot_secao").value != "train":
            falhas.append("conferência: 'Ir para o treino' não abriu Anotação → Treinar")
        n_rev = 2 if outros else 1   # o validado + o reanotado acima
        if not any("1 validated" in s and f"{n_rev} from 🎬 Review" in s
                   for s in textos(at)):
            falhas.append("treino: composição (validadas / Conferência) não apareceu")
        bases = at.selectbox(key="train_base").options
        if "yolo11n.pt (pretrained)" not in bases:
            falhas.append(f"treino: modelos base inesperados {bases}")
    except Exception as e:
        falhas.append(f"{nome}: {type(e).__name__}: {e}")
    return falhas


def main():
    import test_conferencia
    falhas_unid = test_conferencia.rodar_todos()
    print(("ok  " if not falhas_unid else "FAIL") + " seleção de frames sem detecção")
    for f in falhas_unid:
        print("    ", f)
    uid = semear()
    falhou = False
    for lang in ("pt", "en"):
        for pagina in PAGINAS:
            erros, cruas, pt = rodar(uid, lang, pagina)
            estado = "ok  " if not (erros or cruas) else "FAIL"
            falhou |= bool(erros or cruas)
            print(f"{estado} [{lang}] {pagina}")
            for e in erros:
                print("     exceção:", str(e)[:600])
            if cruas:
                print("     chaves cruas na tela:", cruas[:20])
            if pt:
                print("     aviso, português na versão en:", pt[:8])
    falhas_res = checar_resultados(uid)
    falhou |= bool(falhas_res)
    print(("ok  " if not falhas_res else "FAIL") + " minhas análises e conferência")
    for f in falhas_res:
        print("    ", f)
    falhas_nav = checar_navegacao(uid)
    falhou |= bool(falhas_nav)
    print(("ok  " if not falhas_nav else "FAIL") + " navegação e estado entre telas")
    for f in falhas_nav:
        print("    ", f)
    falhas_anot = checar_anotacao_admin(uid)
    falhou |= bool(falhas_anot)
    print(("ok  " if not falhas_anot else "FAIL") + " anotação e administração")
    for f in falhas_anot:
        print("    ", f)
    falhas_conf = checar_conferencia()
    falhou |= bool(falhas_conf) or bool(falhas_unid)
    print(("ok  " if not falhas_conf else "FAIL") + " conferência: anotar frames sem detecção")
    for f in falhas_conf:
        print("    ", f)
    sys.exit(1 if falhou else 0)


if __name__ == "__main__":
    main()
