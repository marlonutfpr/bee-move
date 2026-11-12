"""Seleção dos frames que ficaram sem detecção aceita (tela de Conferência).

Funções puras (sem Streamlit/OpenCV), para decidir QUAIS frames/áreas do
último processamento vão para anotação:

- `frames_aceitos`: por alvo (área ou abelha), os frames com detecção aceita;
- `pares_sem_deteccao`: os pares (frame, alvo) sem detecção aceita;
- `espalhar`: escolhe até N itens distribuídos de forma uniforme e determinística;
- `espalhar_no_tempo`: idem, mas uniforme ao longo do TRECHO (posição do frame).

A leitura dos frames e o passe de baixa confiança ficam em bee_tracker.py
(coletar_frames_abaixo_limiar).
"""


def frames_aceitos(indices_frames, track_ids) -> dict:
    """{alvo: set(frames)} a partir das detecções aceitas da análise."""
    aceitos = {}
    tids = list(track_ids) if track_ids is not None else []
    for i, f in enumerate(indices_frames if indices_frames is not None else []):
        tid = int(tids[i]) if i < len(tids) else 0
        aceitos.setdefault(tid, set()).add(int(f))
    return aceitos


def pares_sem_deteccao(frames_lidos, aceitos, num_alvos, por_area) -> list:
    """Pares (frame, alvo) sem detecção aceita, em ordem de frame.

    Com áreas (`por_area`), o alvo é o índice da área: cada área sem detecção
    aceita naquele frame vira um par. Sem áreas, o alvo é None: entra o frame
    em que foram aceitas menos detecções que `num_alvos` (o frame inteiro).
    """
    num_alvos = max(1, int(num_alvos or 1))
    aceitos = aceitos or {}
    pares = []
    for f in frames_lidos or []:
        f = int(f)
        if por_area:
            pares.extend((f, zi) for zi in range(num_alvos)
                         if f not in aceitos.get(zi, ()))
        elif sum(1 for s in aceitos.values() if f in s) < num_alvos:
            pares.append((f, None))
    return pares


def espalhar(itens, n) -> list:
    """Até `n` itens escolhidos a intervalos iguais (mantém a ordem; determinístico)."""
    itens = list(itens)
    n = int(n)
    if n <= 0:
        return []
    if len(itens) <= n:
        return itens
    if n == 1:
        return [itens[len(itens) // 2]]
    passo = (len(itens) - 1) / (n - 1)
    return [itens[round(i * passo)] for i in range(n)]


def espalhar_no_tempo(pares, frames_lidos, n) -> list:
    """Até `n` pares (frame, alvo) espalhados ao longo do trecho processado.

    Divide o intervalo de frames em `n` faixas iguais e pega um par (o do meio)
    de cada faixa que tem algum; se sobrar vaga, completa com `espalhar` entre
    os não escolhidos. Assim uma longa sequência sem detecção (abelha fora da
    placa por minutos) não toma a amostra inteira. Mantém a ordem; determinístico.
    """
    pares = list(pares)
    n = int(n)
    if n <= 0:
        return []
    if len(pares) <= n:
        return pares
    frames = [int(f) for f in (frames_lidos or [])] or [f for f, _ in pares]
    ini, fim = min(frames), max(frames)
    largura = (fim - ini + 1) / n
    faixas = {}
    for i, (f, _) in enumerate(pares):
        faixas.setdefault(min(n - 1, max(0, int((f - ini) / largura))), []).append(i)
    escolhidos = {g[len(g) // 2] for g in faixas.values()}
    if len(escolhidos) < n:
        resto = [i for i in range(len(pares)) if i not in escolhidos]
        escolhidos.update(espalhar(resto, n - len(escolhidos)))
    return [pares[i] for i in sorted(escolhidos)]


def resumo_faltantes(pares) -> tuple[int, int]:
    """(nº de frames afetados, nº de pares) de uma lista de pares_sem_deteccao."""
    return len({f for f, _ in pares}), len(pares)
