"""Recortes para rotular à mão (a referência do Cerrado).

A referência é feita em recortes das fotos do Mavic 3M, na **resolução
original**, para quem rotula ver todo o detalhe. Cada recorte de rotulação
corresponde a exatamente um recorte do modelo:

    recorte de rotulação: 512×512 pixels da foto original
    recorte do modelo:    256×256 pixels da foto reduzida pela metade

Depois, os polígonos são reduzidos pelo mesmo fator (rsskd/coco.py) e caem
pixel a pixel sobre a entrada do modelo. A imagem mostrada a quem rotula é
a falsa cor (NIR, vermelho, verde: vegetação em vermelho), com as bandas do
próprio Mavic 3M. A foto RGB não serve: vem de outra câmera e fica
desalinhada de 0,3 a 2,6 pixels.

Escolha dos recortes:
  - só de voos que não entram no treino nem na validação (a referência é o
    teste; ver scripts/export_for_labeling.py);
  - fotos seguidas de um voo se sobrepõem (a mesma árvore aparece em
    várias). Por isso os disparos escolhidos ficam a pelo menos `gap`
    números de distância uns dos outros, e cada disparo dá um recorte só;
  - a posição do recorte na foto é sorteada (múltipla de 1/escala, para a
    redução cair na grade do modelo). Tudo com semente fixa: a mesma
    escolha sai igual em qualquer máquina.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cerrado import BAND_ORDER, Capture, find_captures, read_capture

MANIFEST = "manifesto.json"


@dataclass
class Crop:
    """Um recorte de rotulação, em pixels da foto original."""
    arquivo: str                 # nome da imagem gerada para rotular
    voo: str
    disparo: str                 # "<data e hora>_<número>"
    y0: int
    x0: int
    tamanho: int                 # lado, em pixels da foto original
    escala: float                # redução do modelo (0,5: 512 -> 256)


def choose_crops(source: str | Path, flights: list[str], n: int, size: int = 512,
                 scale: float = 0.5, gap: int = 10, seed: int = 0) -> list[Crop]:
    """Sorteia `n` recortes dos voos, um por disparo, com disparos a pelo
    menos `gap` números de distância dentro de cada voo."""
    rng = np.random.default_rng(seed)
    step = round(1 / scale)
    if size % step:
        raise ValueError(f"O tamanho {size} precisa ser múltiplo de {step} (1/escala).")
    candidates: list[Capture] = []
    for f in flights:
        candidates += find_captures(Path(source) / f)
    order = rng.permutation(len(candidates))
    chosen: list[Capture] = []
    for i in order:
        c = candidates[int(i)]
        if all(o.flight != c.flight or abs(int(o.number) - int(c.number)) >= gap for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"Só cabem {len(chosen)} disparos com distância {gap} nesses voos; "
                         "diminua --espaco ou --n, ou acrescente voos.")
    crops = []
    for c in sorted(chosen, key=lambda c: (c.flight, c.key)):
        from PIL import Image

        with Image.open(c.paths["NIR"]) as im:
            w, h = im.size
        if min(w, h) < size:
            raise ValueError(f"Foto {c.key} ({w}×{h}) menor que o recorte de {size}.")
        y0 = int(rng.integers(0, (h - size) // step + 1)) * step
        x0 = int(rng.integers(0, (w - size) // step + 1)) * step
        crops.append(Crop(f"{c.flight}_{c.key}_{y0}_{x0}.png", c.flight, c.key, y0, x0, size,
                          scale))
    return crops


def false_color(bands: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8: NIR, vermelho, verde, cada banda esticada entre os
    percentis 1 e 99 do recorte (só para ver melhor; não muda os dados)."""
    idx = [BAND_ORDER.index(b) for b in ("NIR", "R", "G")]
    out = []
    for i in idx:
        c = bands[i].astype(np.float32)
        lo, hi = np.percentile(c, [1, 99])
        out.append(np.clip((c - lo) / max(hi - lo, 1e-6) * 255, 0, 255))
    return np.dstack(out).astype(np.uint8)


def export_crops(source: str | Path, crops: list[Crop], out_dir: str | Path) -> Path:
    """Grava as imagens em falsa cor e o manifesto.json com a origem de cada uma."""
    from PIL import Image

    out_dir = Path(out_dir)
    (out_dir / "imagens").mkdir(parents=True, exist_ok=True)
    by_flight: dict[str, dict[str, Capture]] = {}
    for crop in crops:
        if crop.voo not in by_flight:
            by_flight[crop.voo] = {c.key: c for c in find_captures(Path(source) / crop.voo)}
        bands = read_capture(by_flight[crop.voo][crop.disparo], scale=1.0)
        piece = bands[:, crop.y0:crop.y0 + crop.tamanho, crop.x0:crop.x0 + crop.tamanho]
        Image.fromarray(false_color(piece)).save(out_dir / "imagens" / crop.arquivo)
    manifest = {"origem": str(source), "recortes": [asdict(c) for c in crops]}
    path = out_dir / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> tuple[str, list[Crop]]:
    """(pasta de origem das fotos, recortes) de um manifesto.json."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return m["origem"], [Crop(**c) for c in m["recortes"]]
