"""Internacionalização (pt/en) da interface.

Uso:
    from i18n import t
    st.button(t("sidebar.logout"))
    st.caption(t("core.progress_status", rel=1, total=9, fps=2.0, n=3))

Os textos ficam em src/locales/<idioma>.json (um dicionário plano chave -> texto).
Chave ausente no idioma escolhido cai para o português e, se nem lá existir,
aparece a própria chave: o app nunca quebra por tradução faltando, e a chave
visível denuncia o que falta (tests/test_i18n.py confere que os dois catálogos
têm as mesmas chaves).

O idioma vem, nesta ordem, de: escolha na sessão, ?lang= na URL (sobrevive ao
F5), variável BEE_LANG, português. Fora de uma sessão
do Streamlit (thread de treino, testes) usa BEE_LANG/português.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

LANGS = {"pt": "Português", "en": "English"}
_DEFAULT = os.environ.get("BEE_LANG", "pt") if os.environ.get("BEE_LANG") in LANGS else "pt"
_DIR = Path(__file__).with_name("locales")


@lru_cache(maxsize=None)
def _load_cached(lang: str, mtime: float) -> dict:
    return json.loads((_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def catalog(lang: str) -> dict:
    """Catálogo do idioma (recarrega se o arquivo mudar)."""
    path = _DIR / f"{lang}.json"
    try:
        return _load_cached(lang, path.stat().st_mtime)
    except FileNotFoundError:
        return {}


def _session():
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return None
        return st
    except Exception:
        return None


def get_lang() -> str:
    st = _session()
    if st is None:
        return _DEFAULT
    lang = st.session_state.get("lang")
    if lang in LANGS:
        return lang
    q = st.query_params.get("lang")
    if q in LANGS:
        st.session_state["lang"] = q
        return q
    return _DEFAULT


def set_lang(lang: str) -> None:
    st = _session()
    if st is None or lang not in LANGS:
        return
    st.session_state["lang"] = lang
    st.query_params["lang"] = lang


def t(key: str, lang: str | None = None, **kw) -> str:
    """Texto traduzido de `key`; `kw` preenche os {placeholders}."""
    lang = lang or get_lang()
    text = catalog(lang).get(key)
    if text is None:
        text = catalog("pt").get(key, key)
    if kw:
        try:
            return text.format(**kw)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def language_selector(container=None, key: str = "lang_selector",
                      disabled: bool = False) -> str:
    """Seletor de idioma (🇧🇷/🇺🇸). Troca o idioma e redesenha a página.

    `disabled`: bloqueia a troca (ex.: durante um processamento, que seria
    interrompido pelo redesenho)."""
    st = _session()
    if st is None:
        return _DEFAULT
    target = container if container is not None else st
    atual = get_lang()
    escolha = target.segmented_control(
        t("common.language"), options=list(LANGS),
        format_func=lambda c: {"pt": "🇧🇷 PT", "en": "🇺🇸 EN"}[c],
        default=atual, key=key, label_visibility="collapsed", required=True,
        disabled=disabled,
    )
    if escolha and escolha != atual:
        set_lang(escolha)
        st.rerun()
    return atual
