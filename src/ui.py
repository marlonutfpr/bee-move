"""Camada de apresentação: tema/CSS global e componentes visuais.

Funções puras de UI (sem lógica de negócio) usadas por bee_tracker.py para dar
um visual profissional, científico e responsivo ao app Streamlit. O tema-base
(cores dos widgets) vem de .streamlit/config.toml; aqui refinamos com CSS e
componentes.

Redesign — "mel sofisticado":
- Tipografia: IBM Plex Sans (interface) + IBM Plex Mono (dados/métricas).
- Paleta âmbar dessaturado + neutros quentes; sobreposições translúcidas
  *quentes* (marrom/âmbar com alpha) — funcionam tanto no tema ESCURO quanto no
  CLARO sem precisar de duas folhas de estilo.
- Cabeçalho com barra de acento (não mais um bloco de gradiente "doce").
- Métricas com valor em fonte mono, hierarquia de rótulo mais clara.
- Vídeos/imagens limitados a uma altura confortável (não estouram a tela).

A API pública é a MESMA de antes — inject_css(), header(title, subtitle),
chip(text, kind) —, então bee_tracker.py não precisa de mudanças para usar.
"""

import html

import streamlit as st

from i18n import t

PRIMARY = "#E3A43A"
PRIMARY_DARK = "#C2862A"
INK = "#1A1714"

# O CSS é agnóstico ao tema: usa sobreposições translúcidas QUENTES (marrom/âmbar
# com alpha) para cartões/bordas, que ficam bem tanto no tema ESCURO (padrão)
# quanto no claro. As cores de fundo/texto dos widgets vêm do tema do Streamlit
# (.streamlit/config.toml), então alternar o tema pelo menu funciona sem quebrar.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bee-primary: #E3A43A;
  --bee-warm-1: rgba(168,132,74,.08);   /* preenchimento de cartão  */
  --bee-warm-2: rgba(168,132,74,.16);   /* borda                    */
  --bee-warm-3: rgba(168,132,74,.24);   /* borda em destaque        */
  --bee-tint:   rgba(227,164,58,.14);   /* tinta âmbar (acento)     */
  --bee-line:   rgba(227,164,58,.45);
  --bee-radius: 12px;
}

html, body, [class*="css"], button, input, textarea, select {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
}

/* dados/números em fonte mono (sensação científica) */
[data-testid="stMetricValue"], code, kbd,
[data-testid="stDataFrame"] [role="gridcell"] {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-feature-settings: "tnum" 1;
}

/* largura confortável; padding-top alto o bastante para o conteúdo NÃO ficar
   sob a barra fixa do Streamlit */
.block-container { padding-top: 4rem; padding-bottom: 3rem; max-width: 1340px; }

/* só esconde o "Made with Streamlit"; o menu ⋮ fica para alternar o tema */
footer { visibility: hidden; }

/* ---- títulos com um pouco mais de presença ---- */
h1, h2, h3 { letter-spacing: -.015em; }

/* =========================================================
   CABEÇALHO — barra de acento + título (sem gradiente pesado)
   ========================================================= */
.bee-header {
  display: flex; align-items: flex-start; gap: 15px; flex-wrap: wrap;
  padding: 2px 0 18px; margin-bottom: 20px;
  border-bottom: 1px solid var(--bee-warm-2);
}
.bee-header .accent {
  width: 4px; align-self: stretch; min-height: 44px; border-radius: 99px;
  background: linear-gradient(180deg, var(--bee-primary), var(--bee-line));
}
.bee-header .mark {
  width: 34px; height: 34px; flex: none; margin-top: 1px;
  background: var(--bee-primary);
  clip-path: polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%);
}
.bee-header .title { margin: 0; font-size: 1.5rem; font-weight: 700; line-height: 1.15; }
.bee-header .subtitle { margin: .3rem 0 0; opacity: .72; font-size: .92rem; }
.bee-header .aside {
  margin-left: auto; align-self: center;
  display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end;
}

/* =========================================================
   MÉTRICAS — cartão com filete de acento + valor mono
   ========================================================= */
[data-testid="stMetric"] {
  background: var(--bee-warm-1);
  border: 1px solid var(--bee-warm-2);
  border-radius: var(--bee-radius);
  padding: 14px 16px 15px;
  position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--bee-primary); opacity: .9;
}
[data-testid="stMetricLabel"] {
  opacity: .68; font-weight: 500; font-size: .80rem;
  letter-spacing: .03em; text-transform: uppercase;
}
[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 600; letter-spacing: -.02em; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; }

/* expanders/containers como cartões */
[data-testid="stExpander"] {
  border: 1px solid var(--bee-warm-2); border-radius: var(--bee-radius); overflow: hidden;
  background: var(--bee-warm-1);
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* botões arredondados, com peso */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 9px; font-weight: 600; border: 1px solid var(--bee-warm-2);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  border-color: transparent;
}

/* abas refinadas: filete de acento na aba ativa */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--bee-warm-2); }
.stTabs [data-baseweb="tab"] {
  border-radius: 8px 8px 0 0; padding: 8px 16px; font-weight: 500;
}
.stTabs [aria-selected="true"] { font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--bee-primary); height: 2px; }

/* radios horizontais como "pílulas" segmentadas */
.stRadio [role="radiogroup"] { gap: 6px; }

/* dataframes/tabelas arredondados */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border-radius: 10px; overflow: hidden; border: 1px solid var(--bee-warm-2);
}

/* divisória sutil + respiro na sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--bee-warm-2); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* mensagens (info/success/warning) um pouco mais discretas */
[data-testid="stAlert"] { border-radius: 10px; }

/* =========================================================
   MÍDIA — vídeos e imagens não estouram mais a tela
   (resolve o "ficam muito grandes na tela"). Centralizados e
   com cantos suaves; a altura máxima mantém tudo na dobra.
   ========================================================= */
[data-testid="stVideo"] video, .stVideo video {
  max-height: 460px; max-width: 100%; width: auto; height: auto;
  border-radius: 10px; border: 1px solid var(--bee-warm-2); display: block;
}
[data-testid="stVideo"] { display: flex; justify-content: center; }

/* imagens (primeiro frame, mapas, gráficos do matplotlib) com teto de altura,
   SEMPRE preservando a proporção. width/height:auto !important vencem o
   width:100% que o st.pyplot/st.image injeta inline — sem isso, o teto de
   altura esmagava os gráficos altos (distorção). object-fit reforça a razão. */
[data-testid="stImage"] img {
  max-height: 520px; max-width: 100%;
  width: auto !important; height: auto !important;
  object-fit: contain; border-radius: 8px;
}
[data-testid="stImage"] { display: flex; justify-content: center; }

/* o canvas de desenho (áreas/calibração) também centralizado e emoldurado */
.streamlit-drawable-canvas, canvas {
  border-radius: 8px;
}

/* responsivo */
@media (max-width: 680px) {
  .block-container { padding-left: .6rem; padding-right: .6rem; }
  .bee-header .title { font-size: 1.2rem; }
  .bee-header .subtitle { font-size: .82rem; }
  [data-testid="stVideo"] video { max-height: 320px; }
}
</style>
"""


def inject_css():
    """Injeta o CSS global. Chamar uma vez por render (idempotente)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title, subtitle="", aside=""):
    """Cabeçalho com barra de acento + marca hexagonal (favo de mel).

    `aside`: HTML opcional alinhado à direita (ex.: chips de modelo/backend).
    """
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    lado = f'<div class="aside">{aside}</div>' if aside else ""
    st.markdown(
        '<div class="bee-header">'
        '<div class="accent"></div>'
        '<div class="mark"></div>'
        f'<div><div class="title">{title}</div>{sub}</div>'
        f'{lado}'
        '</div>',
        unsafe_allow_html=True,
    )


def chip(text, kind="info"):
    """Devolve o HTML de um 'badge' colorido (use com unsafe_allow_html).

    Cores translúcidas/quentes para combinar com ambos os temas (claro/escuro).
    """
    cores = {
        "ok":    ("rgba(58,150,90,.16)",  "#3FAE6B"),
        "warn":  ("rgba(200,140,40,.18)", "#D89A2E"),
        "info":  ("rgba(70,120,210,.16)", "#5E8BD8"),
        "admin": ("rgba(150,100,210,.16)","#9D7BD8"),
        "raw":   ("rgba(215,185,30,.18)", "#C9A417"),   # amarelo (detecção bruta)
    }
    bg, fg = cores.get(kind, cores["info"])
    return (f'<span style="background:{bg};color:{fg};padding:3px 11px;'
            f'border-radius:999px;font-size:.76rem;font-weight:600;'
            f'display:inline-flex;align-items:center;gap:5px;'
            f'white-space:nowrap;">'
            f'<span style="width:6px;height:6px;border-radius:50%;'
            f'background:currentColor;"></span>{text}</span>')


# =========================================================
# Mensagens que sobrevivem ao st.rerun() (toast) e confirmação
# =========================================================

_FLASH_KEY = "_flash"
_FLASH_ICONS = {"success": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌",
                "delete": "🗑️"}


def flash(msg, kind="success"):
    """Guarda uma mensagem para mostrar (toast) na PRÓXIMA execução.

    Use antes de st.rerun(): um st.success() seguido de rerun some antes de o
    usuário ver. `kind`: success, info, warning, error ou delete.
    """
    st.session_state.setdefault(_FLASH_KEY, []).append((str(msg), kind))


def show_flash():
    """Mostra (e consome) as mensagens guardadas por flash(). Chamar 1x por execução."""
    for msg, kind in st.session_state.pop(_FLASH_KEY, None) or []:
        st.toast(msg, icon=_FLASH_ICONS.get(kind, "✅"))


def confirm_delete(title, body_md, on_confirm, done_msg, key):
    """Abre um diálogo de confirmação para uma exclusão irreversível.

    - `body_md`: markdown dizendo o que será excluído;
    - `on_confirm()`: executa a exclusão; devolve None se deu certo ou uma
      mensagem de erro (o diálogo continua aberto e mostra o erro);
    - `done_msg`: toast mostrado após excluir.
    Chame na execução em que o botão "Excluir" foi clicado.
    """
    @st.dialog(title)
    def _dialogo():
        st.markdown(body_md)
        st.warning(t("common.cannot_undo"))
        c_cancel, c_ok = st.columns(2)
        if c_cancel.button(t("common.cancel"), key=f"{key}_cancelar", width="stretch"):
            st.rerun()
        if c_ok.button(t("common.delete_permanently"), key=f"{key}_confirmar",
                       type="primary", width="stretch"):
            erro = on_confirm()
            if erro:
                st.error(erro)
            else:
                flash(done_msg, "delete")
                st.rerun()

    _dialogo()


def esc(text):
    """Escapa texto do usuário/banco antes de ir para HTML (unsafe_allow_html)."""
    return html.escape(str(text))
