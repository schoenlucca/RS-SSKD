"""Rótulos à mão em COCO: polígonos -> uma etiqueta por copa.

CVAT e Roboflow exportam os polígonos no formato COCO: um JSON com as
imagens ("images") e as anotações ("annotations"), cada anotação com o
polígono em "segmentation" ([x1, y1, x2, y2, ...], em pixels da imagem). Este
módulo transforma cada polígono numa copa numerada (0 = fundo, 1..N =
copas), na resolução da imagem rotulada, e depois reduz para a escala do
modelo (rsskd/labeling.py: 512 -> 256).

Detalhes que importam:
  - copas sobrepostas: as anotações são pintadas da maior para a menor, então
    uma copa pequena desenhada por cima de uma grande continua visível;
  - várias partes no mesmo polígono (uma copa cortada por um galho) contam
    como uma copa só;
  - máscaras em RLE (o "pincel" do CVAT) não são lidas: exporte polígonos;
  - o Roboflow renomeia as imagens ("a.png" vira "a_png.rf.<código>.jpg") e
    pode redimensioná-las: o nome original é recuperado, e os polígonos são
    reescalados para o tamanho do recorte;
  - na redução, cada bloco de 2×2 pixels fica com o valor mais comum (copa
    ou fundo), e copas que ficam menores que min_area somem.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_RF = re.compile(r"^(.*)_(png|jpg|jpeg|tif)\.rf\.[0-9a-f]+\.\w+$", re.IGNORECASE)


def original_name(file_name: str) -> str:
    """Nome da imagem como foi exportada (desfaz a renomeação do Roboflow)."""
    name = Path(file_name).name
    m = _RF.match(name)
    return f"{m.group(1)}.{m.group(2).lower()}" if m else name


def rasterize(annotations: list[dict], height: int, width: int,
              scale_xy: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Etiquetas (H, W) int32 a partir das anotações de uma imagem."""
    from skimage.draw import polygon

    shapes = []
    for a in annotations:
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            raise ValueError("Anotação em RLE (máscara de pincel): exporte como polígonos.")
        if not seg:
            continue
        mask = np.zeros((height, width), bool)
        for poly in seg:
            xs = np.asarray(poly[0::2], float) * scale_xy[0]
            ys = np.asarray(poly[1::2], float) * scale_xy[1]
            if len(xs) >= 3:
                rr, cc = polygon(ys, xs, shape=(height, width))
                mask[rr, cc] = True
        if mask.any():
            shapes.append(mask)
    labels = np.zeros((height, width), np.int32)
    for i, mask in enumerate(sorted(shapes, key=lambda m: -int(m.sum())), start=1):
        labels[mask] = i
    return labels


def load_coco(path: str | Path, size: int) -> dict[str, np.ndarray]:
    """{nome original da imagem: etiquetas (size, size)} de um arquivo COCO.

    size: lado do recorte rotulado (512); imagens redimensionadas pela
    ferramenta têm os polígonos reescalados para esse tamanho.
    """
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = {}
    for a in coco.get("annotations", []):
        if not a.get("iscrowd", 0):
            by_image.setdefault(a["image_id"], []).append(a)
    out = {}
    for im in coco["images"]:
        sx, sy = size / im["width"], size / im["height"]
        out[original_name(im["file_name"])] = rasterize(by_image.get(im["id"], []), size,
                                                         size, (sx, sy))
    return out


def downscale_labels(labels: np.ndarray, factor: int, min_area: int = 0) -> np.ndarray:
    """Reduz as etiquetas por `factor` (bloco f×f -> valor mais comum) e
    renumera 1..N, descartando copas com menos de min_area pixels."""
    from scipy import stats

    h, w = labels.shape
    if h % factor or w % factor:
        raise ValueError(f"Tamanho {labels.shape} não é múltiplo de {factor}.")
    blocks = labels.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3)
    small = stats.mode(blocks.reshape(h // factor, w // factor, -1), axis=-1,
                       keepdims=False).mode.astype(np.int32)
    ids, counts = np.unique(small[small > 0], return_counts=True)
    keep = ids[counts >= max(min_area, 1)]
    out = np.zeros_like(small)
    for new, old in enumerate(keep, start=1):
        out[small == old] = new
    return out
