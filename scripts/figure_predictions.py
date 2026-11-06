"""Figura qualitativa para o relatório: as mesmas imagens, vários métodos (parte 7d).

Uso:
    python scripts/figure_predictions.py TEBAKA=data/pseudo/agml \
        "Etapa 1=previsoes/lambda100" "Etapa 2=previsoes/etapa2" --split test
    python scripts/figure_predictions.py TEBAKA=data/pseudo/agml SAM=previsoes/felipe_sam \
        --split test --indices 3 17 42

Cada linha é uma imagem; cada coluna, um método (pastas no formato de
rsskd/predictions.py), e a última, a referência. As copas aparecem como
**contornos coloridos sobre a imagem**, e não como manchas chapadas: assim
dá para ver se o contorno segue a copa de verdade, onde duas copas foram
fundidas e quais copas ficaram de fora. No título de cada painel vão o
número de copas e o F1 daquela imagem contra a referência.

Sem --indices, as imagens são escolhidas espalhadas pela divisão (sempre as
mesmas, para a figura não mudar a cada execução).

Saídas em saidas/: figura_<divisão>.png (200 dpi) e figura_<divisão>.pdf
(vetorial, melhor para o relatório).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances, match_instances  # noqa: E402
from rsskd.predictions import PredictionStore  # noqa: E402


def rgb_image(sample) -> np.ndarray:
    """Imagem RGB (H, W, 3) em [0, 1], com realce de contraste (2% a 98%)."""
    rgb = sample.rgb if sample.rgb is not None else sample.bands[:3].transpose(1, 2, 0)
    rgb = rgb.astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay(rgb: np.ndarray, labels: np.ndarray, color=(1.0, 0.85, 0.0)) -> np.ndarray:
    """Contornos das copas (borda de 1 pixel, por dentro) pintados sobre a imagem."""
    out = rgb.copy()
    out[find_boundaries(labels, mode="inner")] = color
    return out


def parse_methods(items: list[str]) -> list[tuple[str, str]]:
    """'Nome=pasta' -> (nome, pasta); só 'pasta' usa o nome da pasta."""
    methods = []
    for it in items:
        name, _, folder = it.partition("=")
        methods.append((name, folder) if folder else (Path(it).name, it))
    return methods


def figure(ds, methods, split, indices, path_base: Path) -> list[Path]:
    stores = [(name, PredictionStore(folder)) for name, folder in methods]
    cols = len(stores) + 2                                 # imagem + métodos + referência
    fig, ax = plt.subplots(len(indices), cols, figsize=(2.6 * cols, 2.7 * len(indices)),
                           squeeze=False)
    for r, i in enumerate(indices):
        s = ds[i]
        rgb, ref = rgb_image(s), mask_to_instances(s.mask)
        panels = [(rgb, f"imagem {i}")]
        for name, store in stores:
            pred = store.load(split, i)
            f1 = match_instances(pred, ref)["f1"]
            panels.append((overlay(rgb, pred), f"{name}\n{pred.max()} copas, F1 {f1:.2f}"))
        panels.append((overlay(rgb, ref, (0.2, 0.9, 1.0)), f"referência\n{ref.max()} copas"))
        for a, (img, title) in zip(ax[r], panels):
            a.imshow(img)
            a.set_title(title, fontsize=8)
            a.axis("off")
    fig.tight_layout()
    path_base.parent.mkdir(parents=True, exist_ok=True)
    out = [path_base.with_suffix(".png"), path_base.with_suffix(".pdf")]
    fig.savefig(out[0], dpi=200)
    fig.savefig(out[1])
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("metodos", nargs="+", help="Nome=pasta de previsões (ou só a pasta)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--indices", type=int, nargs="*", help="imagens a mostrar")
    ap.add_argument("--n", type=int, default=4, help="quantas imagens, sem --indices")
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    ap.add_argument("--saida", default="saidas")
    args = ap.parse_args(argv)

    ds = AgmlTreeCrowns(args.pasta, split=args.split)
    indices = (args.indices if args.indices
               else np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int).tolist())
    out = figure(ds, parse_methods(args.metodos), args.split, indices,
                 Path(args.saida) / f"figura_{args.split}")
    print(f"Imagens {indices}; figura em {out[0]} e {out[1]}")
    return out


if __name__ == "__main__":
    main()
