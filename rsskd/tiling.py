"""Previsão em imagens grandes (ortomosaicos) por blocos (parte 7e).

O aluno foi treinado em recortes de 256x256, mas um ortomosaico de drone tem
milhares de pixels de lado. A solução usual é:

    1. recortar a imagem em blocos de 256x256 que se sobrepõem (por padrão,
       64 pixels de sobreposição entre vizinhos);
    2. prever as copas em cada bloco, separadamente;
    3. juntar as copas de todos os blocos numa imagem de etiquetas só.

**A costura.** Uma copa perto da divisa entre dois blocos aparece nos dois,
e às vezes cortada num deles. Para cada copa existir uma vez só, e inteira,
cada copa fica com **o bloco cujo centro está mais perto do centro dela**
(o "dono"). Como os blocos se sobrepõem, o dono vê a copa longe da própria
borda, ou seja, inteira; o outro bloco, que talvez a visse cortada,
descarta a sua versão. Esse critério divide o plano sem buracos nem
repetições: todo ponto tem exatamente um bloco mais próximo.

Com 64 pixels de sobreposição, o centro de uma copa fica a pelo menos 32
pixels da borda do bloco dono; copas de até ~64 pixels de diâmetro (as do
AgML têm de ~22 a ~55 de lado) ficam inteiras. Para copas maiores, aumente
a sobreposição.

Tudo aqui recebe a função de previsão de um bloco como parâmetro: serve
para o aluno (model_tile_predictor) e para o professor TEBAKA.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage as ndi

# Recebe as bandas de um bloco (C, tile, tile) e devolve as copas (tile, tile).
TilePredictor = Callable[[np.ndarray], np.ndarray]


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """Posições iniciais dos blocos num eixo, cobrindo [0, size).

    Passo = tile - overlap; o último bloco é encostado no fim da imagem.
    """
    if size <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("A sobreposição tem de ser menor que o bloco.")
    starts = list(range(0, size - tile, step))
    starts.append(size - tile)
    return sorted(set(starts))


def predict_large(bands: np.ndarray, predict_tile: TilePredictor, tile: int = 256,
                  overlap: int = 64, log=None) -> np.ndarray:
    """Copas de uma imagem grande (C, H, W), por blocos sobrepostos.

    Devolve int32 (H, W): 0 = fundo, 1..N = copas, numeradas em ordem.
    Imagens menores que um bloco são completadas com zeros e cortadas de
    volta no fim.
    """
    C, H, W = bands.shape
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    if pad_h or pad_w:
        bands = np.pad(bands, ((0, 0), (0, pad_h), (0, pad_w)))
    _, HP, WP = bands.shape
    ys, xs = tile_starts(HP, tile, overlap), tile_starts(WP, tile, overlap)
    cy = np.array(ys) + tile / 2                       # centros dos blocos
    cx = np.array(xs) + tile / 2

    out = np.zeros((HP, WP), np.int32)
    next_id = 1
    total = len(ys) * len(xs)
    for a, y0 in enumerate(ys):
        for b, x0 in enumerate(xs):
            labels = predict_tile(bands[:, y0:y0 + tile, x0:x0 + tile])
            ids = [i for i in np.unique(labels) if i > 0]
            if ids:
                centers = ndi.center_of_mass(labels > 0, labels, ids)
                for i, (ry, rx) in zip(ids, centers):
                    gy, gx = y0 + ry, x0 + rx              # centro na imagem toda
                    # Só o bloco dono (o de centro mais próximo) fica com a copa.
                    if (int(np.argmin(np.abs(cy - gy))) != a
                            or int(np.argmin(np.abs(cx - gx))) != b):
                        continue
                    region = out[y0:y0 + tile, x0:x0 + tile]
                    paint = (labels == i) & (region == 0)  # sem pisar em outra copa
                    if paint.any():
                        region[paint] = next_id
                        next_id += 1
            if log and (a * len(xs) + b + 1) % 50 == 0:
                log(f"  {a * len(xs) + b + 1}/{total} blocos")
    return out[:H, :W]


def model_tile_predictor(model, spec, device="cpu", score_thr: float = 0.5,
                         mask_thr: float = 0.5) -> TilePredictor:
    """Função de previsão de um bloco com o aluno (Mask R-CNN)."""
    import torch

    from .evaluation import predictions_to_instances

    model.eval()

    def predict(tile_bands: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(spec(tile_bands))).to(device)
        with torch.no_grad():
            (out,) = model([x])
        return predictions_to_instances(out, score_thr, mask_thr)

    return predict


def crown_table(labels: np.ndarray, pixel_size: float | None = None,
                transform=None) -> list[dict]:
    """Uma linha por copa: área (pixels e, com o tamanho do pixel, m²) e
    centro (linha/coluna e, com a georreferência, coordenadas no mapa)."""
    ids = [int(i) for i in np.unique(labels) if i > 0]
    if not ids:
        return []
    areas = ndi.sum(np.ones_like(labels), labels, ids)
    centers = ndi.center_of_mass(labels > 0, labels, ids)
    rows = []
    for i, area, (r, c) in zip(ids, areas, centers):
        row = {"copa": i, "area_px": int(area), "linha": round(float(r), 1),
               "coluna": round(float(c), 1)}
        if pixel_size:
            row["area_m2"] = round(float(area) * pixel_size ** 2, 3)
        if transform is not None:
            col, lin = c + 0.5, r + 0.5                # centro do pixel
            x = transform.a * col + transform.b * lin + transform.c
            y = transform.d * col + transform.e * lin + transform.f
            row["x"], row["y"] = round(float(x), 3), round(float(y), 3)
        rows.append(row)
    return rows
