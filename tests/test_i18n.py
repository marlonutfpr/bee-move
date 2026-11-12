"""Confere os catálogos de tradução (rode: python tests/test_i18n.py).

- pt.json e en.json têm exatamente as mesmas chaves;
- cada texto usa os mesmos {placeholders} nos dois idiomas;
- toda chave literal usada no código (t("...")) existe no catálogo;
- nenhuma chave do catálogo ficou órfã (aviso, não falha).
"""

import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

PT = json.loads((SRC / "locales/pt.json").read_text(encoding="utf-8"))
EN = json.loads((SRC / "locales/en.json").read_text(encoding="utf-8"))
_KEY = re.compile(r"""\bt\(\s*["']([a-z0-9_]+(?:\.[a-z0-9_]+)+)["']""")


def _campos(s: str) -> set[str]:
    try:
        return {f for _, f, _, _ in string.Formatter().parse(s) if f}
    except ValueError:
        return {"<formato inválido>"}


def _chaves_no_codigo() -> set[str]:
    usadas = set()
    for p in SRC.glob("*.py"):
        usadas |= set(_KEY.findall(p.read_text(encoding="utf-8")))
    return usadas


def test_mesmas_chaves():
    assert set(PT) == set(EN), (
        f"só pt: {sorted(set(PT) - set(EN))}; só en: {sorted(set(EN) - set(PT))}")


def test_mesmos_placeholders():
    ruins = [k for k in PT if _campos(PT[k]) != _campos(EN.get(k, ""))]
    assert not ruins, f"placeholders diferentes: {ruins}"


def test_textos_nao_vazios():
    vazios = [k for k, v in {**PT, **EN}.items() if not v.strip()]
    assert not vazios, vazios


def test_chaves_usadas_existem():
    faltando = sorted(_chaves_no_codigo() - set(PT))
    assert not faltando, f"chaves usadas no código sem tradução: {faltando}"


def test_fallback_e_formatacao():
    import i18n
    assert i18n.t("chave.que.nao.existe") == "chave.que.nao.existe"
    assert i18n.t("common.language", lang="en") == "Language"
    assert i18n.t("common.language", lang="pt") == "Idioma"


if __name__ == "__main__":
    falhas = 0
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", nome)
            except AssertionError as e:
                falhas += 1
                print("FAIL", nome, "-", e)
    orfas = sorted(set(PT) - _chaves_no_codigo())
    if orfas:
        print(f"aviso: {len(orfas)} chaves sem uso literal no código "
              f"(podem ser montadas dinamicamente): {orfas[:15]}")
    sys.exit(1 if falhas else 0)
