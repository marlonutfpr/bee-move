"""Geração de dataset no formato YOLO a partir dos candidatos reanotados.

Cada candidato anotado vira uma imagem + um arquivo de rótulos no formato YOLO
(`<classe> <cx> <cy> <w> <h>` normalizado em [0,1]). Candidatos marcados como
"sem abelha" viram amostras negativas (arquivo de rótulo vazio). A estrutura
gerada é compatível com Ultralytics (data.yaml + images/labels em train/val),
pronta para o fine-tuning posterior.

Função pura: recebe os candidatos (dicts vindos do banco, com `image_jpg`,
`labels`, `width`, `height`) e devolve os bytes de um .zip.
"""

import io
import json
import os
import zipfile

from i18n import t


def _linhas_yolo(labels, width, height, class_id=0):
    """Converte caixas [x, y, w, h] (px, canto sup-esq) em linhas YOLO normalizadas."""
    linhas = []
    if not width or not height:
        return linhas
    for box in labels or []:
        x, y, w, h = box[:4]
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        nw, nh = w / width, h / height
        # descarta caixas degeneradas e recorta ao quadro
        if nw <= 0 or nh <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(nw, 1.0), min(nh, 1.0)
        linhas.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return linhas


def _parse(valor):
    if isinstance(valor, (list, tuple)):
        return valor
    try:
        return json.loads(valor) if valor else []
    except (ValueError, TypeError):
        return []


def build_yolo_zip(candidatos, class_name="Abelha") -> bytes:
    """Empacota os candidatos anotados num .zip YOLO (Ultralytics-friendly).

    Estrutura:
        images/train/<nome>.jpg
        labels/train/<nome>.txt   (vazio = amostra negativa)
        data.yaml
        classes.txt
        README.txt
    """
    buf = io.BytesIO()
    n_imgs = n_caixas = n_neg = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in candidatos:
            jpg = c.get("image_jpg")
            if not jpg:
                continue
            nome = f"cand{c.get('id')}_f{c.get('frame_index') or 0}"
            labels = _parse(c.get("labels"))
            linhas = _linhas_yolo(labels, c.get("width"), c.get("height"))
            zf.writestr(f"images/train/{nome}.jpg", jpg)
            zf.writestr(f"labels/train/{nome}.txt", "\n".join(linhas))
            n_imgs += 1
            n_caixas += len(linhas)
            if not linhas:
                n_neg += 1

        data_yaml = (
            "path: .\n"
            "train: images/train\n"
            "val: images/train\n"
            "nc: 1\n"
            f"names:\n  0: {class_name}\n"
        )
        zf.writestr("data.yaml", data_yaml)
        zf.writestr("classes.txt", f"{class_name}\n")
        zf.writestr("README.txt",
                    t("dataset.readme", n_imgs=n_imgs, n_boxes=n_caixas, n_neg=n_neg))
    return buf.getvalue()


def export_to_dir(candidatos, dest, class_name="Abelha", extra_train_dirs=None):
    """Escreve o dataset YOLO em disco (para o treino Ultralytics).

    Cria dest/images/train, dest/labels/train e dest/data.yaml. `extra_train_dirs`
    são pastas de imagens YOLO adicionais (ex.: dataset externo importado) que
    entram no treino junto com os candidatos. Devolve
    (caminho_data_yaml, n_imagens_candidatos, n_caixas_candidatos).
    """
    img_dir = os.path.join(dest, "images", "train")
    lbl_dir = os.path.join(dest, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n_imgs = n_caixas = 0
    for c in candidatos:
        jpg = c.get("image_jpg")
        if not jpg:
            continue
        nome = f"cand{c.get('id')}_f{c.get('frame_index') or 0}"
        linhas = _linhas_yolo(_parse(c.get("labels")), c.get("width"), c.get("height"))
        with open(os.path.join(img_dir, nome + ".jpg"), "wb") as f:
            f.write(jpg)
        with open(os.path.join(lbl_dir, nome + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))
        n_imgs += 1
        n_caixas += len(linhas)

    # Lista de pastas de imagens p/ treino (candidatos + extras). Caminhos
    # absolutos; o Ultralytics acha os labels trocando 'images' por 'labels'.
    train_dirs = [os.path.abspath(img_dir)]
    for d in (extra_train_dirs or []):
        if os.path.isdir(d):
            train_dirs.append(os.path.abspath(d))

    data_yaml = os.path.join(dest, "data.yaml")
    linhas = [f"path: {os.path.abspath(dest)}", "train:"]
    linhas += [f"  - {d}" for d in train_dirs]
    linhas += ["val:"]
    linhas += [f"  - {d}" for d in train_dirs]
    linhas += ["nc: 1", "names:", f"  0: {class_name}"]
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    return data_yaml, n_imgs, n_caixas


def resumo(candidatos) -> dict:
    """Contagens para mostrar antes de exportar."""
    n_imgs = n_caixas = n_neg = 0
    for c in candidatos:
        if not c.get("image_jpg"):
            continue
        n_imgs += 1
        linhas = _linhas_yolo(_parse(c.get("labels")), c.get("width"), c.get("height"))
        n_caixas += len(linhas)
        if not linhas:
            n_neg += 1
    return {"imagens": n_imgs, "caixas": n_caixas, "negativas": n_neg}
