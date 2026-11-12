"""Seleção dos frames sem detecção da Conferência (rode: python tests/test_conferencia.py).

Funções puras de src/conferencia.py — não precisam de Streamlit nem OpenCV.
Também são chamadas pelo tests/smoke_app.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import conferencia  # noqa: E402


def test_frames_aceitos():
    aceitos = conferencia.frames_aceitos([0, 0, 5, 10], [0, 1, 0, 1])
    assert aceitos == {0: {0, 5}, 1: {0, 10}}, aceitos
    # sem track_ids (análises antigas): tudo é do alvo 0
    assert conferencia.frames_aceitos([3, 4], None) == {0: {3, 4}}
    assert conferencia.frames_aceitos([], []) == {}


def test_pares_por_area():
    aceitos = {0: {0, 5}, 1: {0, 10}}
    pares = conferencia.pares_sem_deteccao([0, 5, 10, 15], aceitos, 2, True)
    assert pares == [(5, 1), (10, 0), (15, 0), (15, 1)], pares
    # área sem nenhuma detecção na análise: todos os frames entram
    assert conferencia.pares_sem_deteccao([0, 5], {}, 1, True) == [(0, 0), (5, 0)]


def test_pares_frame_inteiro():
    aceitos = {0: {0, 5}, 1: {0, 10}}
    # 2 abelhas esperadas: entra o frame com menos de 2 detecções aceitas
    pares = conferencia.pares_sem_deteccao([0, 5, 10, 15], aceitos, 2, False)
    assert pares == [(5, None), (10, None), (15, None)], pares
    # 1 abelha: só o frame sem nenhuma detecção
    assert conferencia.pares_sem_deteccao([0, 5, 10, 15], aceitos, 1, False) == [(15, None)]


def test_espalhar():
    assert conferencia.espalhar(range(10), 3) == [0, 4, 9]
    assert conferencia.espalhar(range(3), 5) == [0, 1, 2]
    assert conferencia.espalhar([], 3) == []
    assert conferencia.espalhar(range(10), 0) == []
    assert conferencia.espalhar(range(9), 1) == [4]
    sel = conferencia.espalhar(range(1000), 200)
    assert len(sel) == len(set(sel)) == 200 and sel[0] == 0 and sel[-1] == 999
    assert sel == conferencia.espalhar(range(1000), 200)   # determinístico


def test_espalhar_no_tempo():
    # 0..999 lidos; uma sequência longa sem detecção (100..899) + 10 avulsos
    avulsos = [(f, None) for f in range(905, 1000, 10)]
    pares = [(f, None) for f in range(100, 900)] + avulsos
    sel = conferencia.espalhar_no_tempo(pares, range(1000), 20)
    assert len(sel) == len(set(sel)) == 20, sel
    assert sel == sorted(sel)
    # uniforme no tempo: a sequência longa (80% do trecho) não toma tudo
    assert sum(1 for p in sel if p in avulsos) >= 2, sel
    assert sum(1 for f, _ in sel if f < 900) <= 18, sel
    assert sel == conferencia.espalhar_no_tempo(pares, range(1000), 20)
    # poucos pares: devolve todos; n <= 0: nada
    assert conferencia.espalhar_no_tempo(avulsos, range(1000), 50) == avulsos
    assert conferencia.espalhar_no_tempo(pares, range(1000), 0) == []
    # pares por área (vários no mesmo frame) e sem frames_lidos
    por_area = [(f, zi) for f in range(0, 100, 5) for zi in (0, 1)]
    sel = conferencia.espalhar_no_tempo(por_area, None, 10)
    assert len(sel) == len(set(sel)) == 10 and set(sel) <= set(por_area), sel


def test_resumo_faltantes():
    assert conferencia.resumo_faltantes([(5, 1), (10, 0), (10, 1)]) == (2, 3)
    assert conferencia.resumo_faltantes([]) == (0, 0)


def rodar_todos() -> list[str]:
    """Roda os test_* e devolve as falhas (usado pelo smoke_app)."""
    falhas = []
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                falhas.append(f"{nome}: {e}")
    return falhas


if __name__ == "__main__":
    falhas = rodar_todos()
    for f in falhas:
        print("FAIL", f)
    print("ok   conferencia" if not falhas else f"{len(falhas)} falha(s)")
    sys.exit(1 if falhas else 0)
