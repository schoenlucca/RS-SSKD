"""Desenha um lote do aluno pronto, para conferir a olho que tudo se alinha.

Uso:
    python scripts/show_batch.py
    python scripts/show_batch.py --split val --n 6 --semente 3

Pega imagens do CrownDataset exatamente como o treino vai recebê-las (já
normalizadas, já no formato do Mask R-CNN) e desenha, para cada uma:

    1. a imagem com as copas do professor (máscaras coloridas e caixas);
    2. a mesma imagem depois de uma rotação/espelhamento sorteado, com as
       copas transformadas junto;
    3. o canal de NDVI que o aluno recebe;
    4. a referência do conjunto (só para comparar; o aluno não a vê).

O que conferir na figura:
  - nas colunas 1 e 2, cada máscara e cada caixa estão em cima de uma copa
    da imagem, e não deslocadas ou giradas em relação a ela;
  - a coluna 2 mostra a mesma cena da coluna 1, só que girada/espelhada;
  - as cores da imagem estão naturais: a desnormalização funcionou.

Um erro de alinhamento aqui não daria nenhuma mensagem de erro no treino:
o aluno só aprenderia errado. Por isso vale olhar antes de treinar.

Saída: saidas/lote_<divisão>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.augment import RandomDihedral, make_loader  # noqa: E402
from rsskd.data import DEFAULT_DIR  # noqa: E402
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.inputs import InputSpec  # noqa: E402


def rgb_from_input(image: torch.Tensor, spec: InputSpec) -> np.ndarray:
    """Imagem (H, W, 3) em [0, 1] a partir da entrada normalizada do aluno."""
    x = spec.denormalize(image.numpy())
    names = spec.channels
    idx = ([names.index(c) for c in ("vermelho", "verde", "azul")]
           if all(c in names for c in ("vermelho", "verde", "azul")) else [0, 1, 2])
    return np.clip(x[idx].transpose(1, 2, 0) / 255.0, 0, 1)


def draw_crowns(ax, rgb: np.ndarray, target: dict, title: str, seed: int = 0):
    """Imagem com as máscaras (coloridas, semitransparentes) e as caixas."""
    masks = target["masks"].numpy().astype(bool)
    colors = np.random.default_rng(seed).random((len(masks), 3))
    over = rgb.copy()
    for m, c in zip(masks, colors):
        over[m] = 0.5 * over[m] + 0.5 * c
    ax.imshow(over)
    for (x0, y0, x1, y1), c in zip(target["boxes"].tolist(), colors):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=c, linewidth=1))
    ax.set_title(f"{title}: {len(masks)} copas", fontsize=9)


def figure(plain: CrownDataset, augmented: CrownDataset, indices, path):
    """Monta a figura de 4 colunas para as imagens em `indices`."""
    spec = plain.spec
    ndvi_ch = spec.channels.index("ndvi") if "ndvi" in spec.channels else -1
    loader_plain = make_loader(torch.utils.data.Subset(plain, indices),
                               batch_size=len(indices), shuffle=False)
    loader_aug = make_loader(torch.utils.data.Subset(augmented, indices),
                             batch_size=len(indices), shuffle=False)
    imgs, targets = next(iter(loader_plain))      # o mesmo caminho do treino
    imgs_a, targets_a = next(iter(loader_aug))

    fig, ax = plt.subplots(len(indices), 4, figsize=(13, 3.3 * len(indices)),
                           squeeze=False)
    for r, i in enumerate(indices):
        draw_crowns(ax[r, 0], rgb_from_input(imgs[r], spec), targets[r],
                    f"imagem {i}, professor")
        draw_crowns(ax[r, 1], rgb_from_input(imgs_a[r], spec), targets_a[r],
                    "girada/espelhada")
        ndvi = spec.denormalize(imgs[r].numpy())[ndvi_ch]
        im = ax[r, 2].imshow(ndvi, cmap="RdYlGn")
        ax[r, 2].set_title("canal NDVI do aluno", fontsize=9)
        fig.colorbar(im, ax=ax[r, 2], fraction=0.046)
        ref = plain.reference(i)
        colors = np.random.default_rng(1).random((ref.max() + 1, 3))
        colors[0] = 0
        ax[r, 3].imshow(colors[ref])
        ax[r, 3].set_title(f"referência: {ref.max()} copas", fontsize=9)
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    ap.add_argument("--copas", default="data/pseudo/agml",
                    help="pasta das copas do professor")
    ap.add_argument("--entrada", default="configs/agml_entrada.json")
    ap.add_argument("--n", type=int, default=4, help="quantas imagens")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.semente)
    spec = InputSpec.load(args.entrada)
    plain = CrownDataset(args.split, args.copas, spec, dataset_dir=args.pasta)
    augmented = CrownDataset(args.split, args.copas, spec, dataset_dir=args.pasta,
                             transform=RandomDihedral())
    rng = np.random.default_rng(args.semente)
    indices = sorted(rng.choice(len(plain), size=min(args.n, len(plain)),
                                replace=False).tolist())
    out = args.saida or f"saidas/lote_{args.split}.png"
    figure(plain, augmented, indices, out)
    print(f"Imagens {indices} desenhadas em {out}")
    return out


if __name__ == "__main__":
    main()
