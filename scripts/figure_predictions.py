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

Cerrado. As fotos do Mavic 3M não têm RGB alinhado: com --falsa-cor e a
configuração do professor (configs/cerrado_teacher.json), o fundo vira a
falsa cor NIR, vermelho, verde (vegetação em vermelho), com as bandas lidas
da seção "bandas". Enquanto não há referência (máscaras vazias), a coluna da
referência e o F1 saem da figura, e ela compara só os métodos entre si:
    python3 scripts/figure_predictions.py TEBAKA=data/pseudo/cerrado \
        "Aluno=previsoes/cerrado_l0_s0" --pasta data/cerrado --split val \
        --falsa-cor configs/cerrado_teacher.json

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
from rsskd.paths import HINT_FOLDER, require  # noqa: E402
from rsskd.predictions import PredictionStore  # noqa: E402


def rgb_image(sample, bands: tuple[int, int, int] | None = None) -> np.ndarray:
    """Imagem (H, W, 3) em [0, 1], com realce de contraste (2% a 98%).

    bands: None = o RGB do conjunto (ou as 3 primeiras bandas); três índices =
    essas bandas como vermelho, verde e azul da figura (falsa cor), cada uma
    realçada separadamente.
    """
    if bands is not None:
        chans = [sample.bands[b].astype(np.float32) for b in bands]
        out = []
        for c in chans:
            lo, hi = np.percentile(c, 2), np.percentile(c, 98)
            out.append(np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1))
        return np.dstack(out)
    rgb = sample.rgb if sample.rgb is not None else sample.bands[:3].transpose(1, 2, 0)
    rgb = rgb.astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def false_color_bands(config_path) -> tuple[int, int, int]:
    """Índices de NIR, vermelho e verde na seção "bandas" da configuração."""
    from rsskd.teacher import load_config

    b = load_config(config_path)["bandas"]
    return b["nir"], b["vermelho"], b["verde"]


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


def figure(ds, methods, split, indices, path_base: Path,
           bands: tuple[int, int, int] | None = None) -> list[Path]:
    stores = [(name, PredictionStore(require(folder, f"a pasta de previsões de {name}",
                                             HINT_FOLDER)))
              for name, folder in methods]
    refs = {i: mask_to_instances(ds[i].mask) for i in indices}
    has_ref = any(r.max() > 0 for r in refs.values())     # sem referência: não há F1
    cols = len(stores) + 1 + int(has_ref)                 # imagem + métodos (+ referência)
    fig, ax = plt.subplots(len(indices), cols, figsize=(2.6 * cols, 2.7 * len(indices)),
                           squeeze=False)
    for r, i in enumerate(indices):
        s = ds[i]
        rgb, ref = rgb_image(s, bands), refs[i]
        panels = [(rgb, f"imagem {i}")]
        for name, store in stores:
            pred = store.load(split, i)
            title = f"{name}\n{pred.max()} copas"
            if has_ref:
                title += f", F1 {match_instances(pred, ref)['f1']:.2f}"
            panels.append((overlay(rgb, pred), title))
        if has_ref:
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
    ap.add_argument("--falsa-cor", default=None, metavar="CONFIG",
                    help="usa a falsa cor NIR, vermelho, verde, com as bandas da configuração "
                         "do professor (para conjuntos sem RGB, como o Cerrado)")
    args = ap.parse_args(argv)

    ds = AgmlTreeCrowns(args.pasta, split=args.split)
    indices = (args.indices if args.indices
               else np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int).tolist())
    bands = false_color_bands(args.falsa_cor) if args.falsa_cor else None
    out = figure(ds, parse_methods(args.metodos), args.split, indices,
                 Path(args.saida) / f"figura_{args.split}", bands)
    print(f"Imagens {indices}; figura em {out[0]} e {out[1]}")
    return out


if __name__ == "__main__":
    main()
