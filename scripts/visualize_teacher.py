"""Roda o professor TEBAKA numa imagem e salva uma figura para inspeção.

Uso (imagem real, GeoTIFF com várias bandas):
    python scripts/visualize_teacher.py caminho/imagem.tif --nir 3 --red 1

Uso (imagem sintética, para conferir que tudo funciona):
    python scripts/visualize_teacher.py --sintetica

A figura é salva em saidas/ (ignorada pelo git) e mostra: a imagem, o NDVI, a máscara de vegetação e as copas
separadas (cada copa com uma cor). Use-a para ajustar os limiares
(--th1, --th2) e o --min-distance olhando o resultado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.teacher import TebakaTeacher  # noqa: E402


def read_image(path: str) -> np.ndarray:
    """Lê um GeoTIFF multibanda e devolve (H, W, C) em float32."""
    try:
        import rasterio  # type: ignore[import-not-found]  # opcional
    except ImportError:
        try:
            import tifffile
        except ImportError:
            raise SystemExit("Instale rasterio ou tifffile: pip install rasterio")
        img = tifffile.imread(path).astype(np.float32)
        # tifffile costuma devolver (C, H, W) para imagens multibanda.
        if img.ndim == 3 and img.shape[0] <= 16:
            img = np.moveaxis(img, 0, -1)
        return img
    with rasterio.open(path) as src:
        print(f"{path}: {src.count} bandas, tipo {src.dtypes[0]}, "
              f"descrições {src.descriptions}")
        return np.moveaxis(src.read().astype(np.float32), 0, -1)


def rgb_preview(img: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """Composição colorida com esticamento de contraste (percentis 2-98)."""
    out = img[..., list(rgb)].copy()
    for c in range(3):
        lo, hi = np.percentile(out[..., c], [2, 98])
        out[..., c] = np.clip((out[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagem", nargs="?")
    ap.add_argument("--sintetica", action="store_true")
    ap.add_argument("--nir", type=int, default=3)
    ap.add_argument("--red", type=int, default=1)
    ap.add_argument("--rgb", type=int, nargs=3, default=None,
                    help="índices das bandas para a composição colorida")
    ap.add_argument("--th1", type=float, default=0.4)
    ap.add_argument("--th2", type=float, default=0.4)
    ap.add_argument("--min-distance", type=int, default=15)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--saida", default="saidas/professor_tebaka.png")
    args = ap.parse_args()

    if args.sintetica:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from test_teacher import synthetic_image
        img, _ = synthetic_image()
        args.nir, args.red = 3, 1
    elif args.imagem:
        img = read_image(args.imagem)
    else:
        ap.error("informe o caminho da imagem ou --sintetica")

    teacher = TebakaTeacher(nir_band=args.nir, red_band=args.red,
                            th1=args.th1, th2=args.th2,
                            min_distance=args.min_distance,
                            min_area=args.min_area)
    out = teacher(img)
    print(f"Copas encontradas: {out['n_crowns']}")

    rgb = args.rgb or ([args.red, args.nir, args.nir] if img.shape[-1] < 3
                       else [args.red, min(2, img.shape[-1] - 1), 0])
    inst = out["instances"]
    colors = np.random.default_rng(0).random((inst.max() + 1, 3))
    colors[0] = 0
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    r, g, b = rgb
    ax[0].imshow(rgb_preview(img, (r, g, b)))
    ax[0].set_title("Imagem (composição)")
    im = ax[1].imshow(out["ndvi"], cmap="RdYlGn", vmin=-1, vmax=1)
    ax[1].set_title("NDVI")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(out["vegetation"], cmap="gray")
    ax[2].set_title("Máscara de vegetação")
    ax[3].imshow(colors[inst])
    ax[3].set_title(f"Copas separadas: {out['n_crowns']}")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.saida, dpi=120)
    print(f"Figura salva em {args.saida}")


if __name__ == "__main__":
    main()
