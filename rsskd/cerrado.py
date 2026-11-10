"""Leitura das fotos multiespectrais do Cerrado (DJI Mavic 3M).

Cada disparo do Mavic 3M gera uma foto por banda multiespectral. Nas pastas
do laboratório (projeto_cerrado/<voo>/), elas já vêm processadas:

    processed-DJI_20250616115856_0028_MS_G.TIF     verde
    processed-DJI_20250616115856_0028_MS_R.TIF     vermelho
    processed-DJI_20250616115856_0028_MS_RE.TIF    Red-Edge
    processed-DJI_20250616115856_0028_MS_NIR.TIF   infravermelho próximo

O que medimos nelas (08/11/2026):
  - as 4 bandas de um disparo têm 1000×1000 pixels e estão alinhadas entre
    si (deslocamento de 0,0 pixel, medido por correlação de fase);
  - os valores são de 8 bits, numa conversão fixa para todas as fotos (o
    vermelho nunca fica abaixo de ~20), não reflectância;
  - a foto RGB (D.JPG e a composição RGB_NDRE) vem de outra câmera e fica
    desalinhada de 0,3 a 2,6 pixels conforme a região. Por isso a banda
    azul não é usada: a entrada fica com as 4 bandas do Mavic 3M + NDVI.

Este módulo transforma esses disparos no mesmo formato do conjunto AgML
(rsskd/data.py): arquivos Parquet com recortes de 256×256 e as bandas em
uint8 (C, H, W). Assim o resto do código (professor, normalização, treino,
avaliação) roda no Cerrado sem mudança, só com outra --pasta e outra
configuração de bandas (configs/cerrado_teacher.json).

Escala. Nas fotos de 1000 pixels, as copas grandes têm de 150 a 250 pixels
de diâmetro: num recorte de 256, quase todas sairiam cortadas na borda. Por
isso a foto é reduzida antes de recortar (padrão: pela metade, para
500×500, que dá 4 recortes de 256 por disparo, com 12 pixels de
sobreposição). A redução usa a média dos pixels (Image.BOX), que não cria
serrilhado.

Divisão. Fotos seguidas de um voo se sobrepõem (a mesma árvore aparece em
várias). Por isso a divisão é **por voo**, nunca por foto: um voo inteiro vai
para treino, validação ou teste.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tiling import tile_starts

# Ordem das bandas nos recortes (e índices em configs/cerrado_teacher.json).
BAND_ORDER = ("R", "G", "RE", "NIR")
BAND_NAMES = {"R": "vermelho", "G": "verde", "RE": "red_edge", "NIR": "nir"}

_NAME = re.compile(r"DJI_(\d{14})_(\d{4})_MS_(G|R|RE|NIR)\.TIF$", re.IGNORECASE)


@dataclass
class Capture:
    """Um disparo: as 4 bandas de uma foto."""
    flight: str
    key: str                     # "<data e hora>_<número>", ex.: 20250616115856_0028
    paths: dict[str, Path]       # banda -> arquivo

    @property
    def number(self) -> str:
        return self.key.split("_")[1]


def find_captures(flight_dir: str | Path) -> list[Capture]:
    """Os disparos completos (com as 4 bandas) de uma pasta de voo, em ordem.

    Disparos com banda faltando são ignorados (e contados no aviso).
    """
    flight_dir = Path(flight_dir)
    if not flight_dir.is_dir():
        raise FileNotFoundError(f"Não achei a pasta do voo: {flight_dir}")
    found: dict[str, dict[str, Path]] = {}
    for f in flight_dir.iterdir():
        m = _NAME.search(f.name)
        if m:
            found.setdefault(f"{m.group(1)}_{m.group(2)}", {})[m.group(3).upper()] = f
    complete = [Capture(flight_dir.name, k, p) for k, p in sorted(found.items())
                if set(p) == set(BAND_ORDER)]
    missing = len(found) - len(complete)
    if missing:
        print(f"Aviso: {missing} disparos de {flight_dir.name} sem as 4 bandas foram ignorados.")
    return complete


def read_capture(capture: Capture, scale: float = 0.5) -> np.ndarray:
    """As 4 bandas de um disparo, (4, H, W) uint8, na ordem de BAND_ORDER,
    reduzidas por `scale` (1 = tamanho original)."""
    from PIL import Image

    bands = []
    for b in BAND_ORDER:
        with Image.open(capture.paths[b]) as im:
            if im.mode not in ("L", "I;16", "I"):
                im = im.convert("L")
            if scale != 1:
                size = (round(im.width * scale), round(im.height * scale))
                # Image.Resampling só existe a partir do Pillow 9.1.
                box = getattr(getattr(Image, "Resampling", Image), "BOX")
                im = im.resize(size, box)
            a = np.asarray(im)
        if a.dtype != np.uint8:
            raise ValueError(f"{capture.paths[b].name}: esperava 8 bits, veio {a.dtype}.")
        bands.append(a)
    shapes = {a.shape for a in bands}
    if len(shapes) != 1:
        raise ValueError(f"Disparo {capture.key}: bandas com tamanhos diferentes {shapes}.")
    return np.stack(bands)


def tiles(image: np.ndarray, tile: int = 256, overlap: int = 12):
    """Recortes (y0, x0, recorte) de uma imagem (C, H, W), cobrindo toda ela."""
    _, h, w = image.shape
    for y0 in tile_starts(h, tile, overlap):
        for x0 in tile_starts(w, tile, overlap):
            yield y0, x0, image[:, y0:y0 + tile, x0:x0 + tile]


def _png(a: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def capture_rows(capture: Capture, split: str, scale: float = 0.5, tile: int = 256,
                 overlap: int = 12) -> list[dict]:
    """As linhas do Parquet (formato AgML) com os recortes de um disparo.

    A coluna mask vai vazia (zeros): no Cerrado, a referência só existe nas
    imagens rotuladas à mão, que entram depois.
    """
    image = read_capture(capture, scale)
    rows = []
    empty_mask = _png(np.zeros((tile, tile), np.uint8))
    for y0, x0, t in tiles(image, tile, overlap):
        if t.shape[1:] != (tile, tile):
            raise ValueError(f"Disparo {capture.key} menor que um recorte ({t.shape}).")
        meta = {"voo": capture.flight, "disparo": capture.key, "y0": y0, "x0": x0,
                "escala": scale, "recorte": tile,
                "arquivo": capture.paths["NIR"].name}
        rows.append({"bands": t.tobytes(), "bands_shape": list(t.shape),
                     "bands_dtype": "uint8", "band_metadata": json.dumps(meta),
                     "band_order": json.dumps([BAND_NAMES[b] for b in BAND_ORDER]),
                     "mask": empty_mask, "rgb": None, "split": split})
    return rows


def build_dataset(source: str | Path, out_dir: str | Path, splits: dict[str, list[str]],
                  scale: float = 0.5, tile: int = 256, overlap: int = 12, every: int = 1,
                  log=print) -> dict:
    """Grava out_dir/<divisão>.parquet a partir das pastas de voo.

    splits: {"train": ["2025-01-27-voo1", ...], "val": [...], "test": [...]}.
    every:  usa um disparo a cada `every` (fotos seguidas se sobrepõem muito;
            2 ou 3 reduz a redundância sem perder área).
    Devolve o resumo (também gravado em out_dir/manifesto.json).
    """
    import pandas as pd

    flights = [f for fs in splits.values() for f in fs]
    repeated = {f for f in flights if flights.count(f) > 1}
    if repeated:
        raise ValueError(f"Voos em mais de uma divisão: {sorted(repeated)}. "
                         "A divisão é por voo: cada voo vai para uma só.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"origem": str(source), "escala": scale, "recorte": tile,
               "sobreposicao": overlap, "a_cada": every,
               "bandas": [BAND_NAMES[b] for b in BAND_ORDER], "divisoes": {}}
    for split, names in splits.items():
        rows, per_flight = [], {}
        for name in names:
            caps = find_captures(Path(source) / name)[::every]
            n0 = len(rows)
            for i, c in enumerate(caps):
                rows += capture_rows(c, split, scale, tile, overlap)
                if log and (i + 1) % 50 == 0:
                    log(f"  {split}/{name}: {i + 1}/{len(caps)} disparos")
            per_flight[name] = {"disparos": len(caps), "recortes": len(rows) - n0}
            if log:
                log(f"{split}: {name}, {len(caps)} disparos, {len(rows) - n0} recortes")
        if rows:
            pd.DataFrame(rows).to_parquet(out_dir / f"{split}.parquet", index=False)
        summary["divisoes"][split] = {"voos": per_flight, "recortes": len(rows)}
    (out_dir / "manifesto.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return summary
