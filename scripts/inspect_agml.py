"""Baixa e inspeciona o conjunto de copas do Hugging Face (Project-AgML).

Uso:
    python scripts/inspect_agml.py --baixar      # baixa (~330 MB) e inspeciona
    python scripts/inspect_agml.py               # só inspeciona (já baixado)

O que ele mostra:
  1. As colunas dos arquivos Parquet e o tipo de cada uma (para conferir se
     o carregador rsskd/data.py entendeu o formato).
  2. Quantas amostras há em cada divisão (train, val, test).
  3. Os metadados e o formato da primeira amostra.
  4. A razão copa/fundo de cada banda e o palpite de qual é o NIR e qual é o
     vermelho.
  5. Uma figura de cada banda da primeira amostra, com a máscara, salva em
     saidas/agml_bandas.png.
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
from rsskd.data import (  # noqa: E402
    DEFAULT_DIR, AgmlTreeCrowns, download_agml, guess_nir_red, parquet_files,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baixar", action="store_true")
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    args = ap.parse_args()

    if args.baixar:
        print(f"Baixando para {args.pasta} ...")
        download_agml(args.pasta)

    # 1. Colunas e tipos.
    import pyarrow.parquet as pq
    files = parquet_files(args.pasta)
    print(f"\nArquivos: {[f.name for f in files]}")
    print("Colunas e tipos:")
    for field in pq.read_schema(files[0]):
        print(f"  {field.name}: {field.type}")

    # 2. Divisões.
    ds = AgmlTreeCrowns(args.pasta)
    print(f"\nTotal de amostras: {len(ds)}")
    if "split" in ds.df.columns:
        print(ds.df["split"].value_counts().to_string())

    # 3. Primeira amostra.
    s = ds[0]
    print(f"\nPrimeira amostra: bandas {s.bands.shape} {s.bands.dtype}, "
          f"máscara {s.mask.shape}, "
          f"rgb {None if s.rgb is None else s.rgb.shape}, split '{s.split}'")
    print(f"Fração de copa na máscara: {s.mask.mean():.0%}")
    print(f"Metadados: {str(s.metadata)[:600]}")
    print("Valores por banda (mín, média, máx):")
    for c in range(s.bands.shape[0]):
        b = s.bands[c]
        print(f"  banda {c}: {b.min():.0f}, {b.mean():.1f}, {b.max():.0f}")

    # 4. Palpite de NIR e vermelho.
    nir, red, ratios = guess_nir_red(ds)
    print("\nRazão média copa/fundo por banda:")
    for c, r in enumerate(ratios):
        print(f"  banda {c}: {r:.2f}")
    print(f"Palpite: NIR = banda {nir}, vermelho = banda {red}")

    # 5. Figura das bandas.
    n = s.bands.shape[0]
    fig, ax = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.4))
    for c in range(n):
        ax[c].imshow(s.bands[c], cmap="gray")
        ax[c].set_title(f"banda {c} (razão {ratios[c]:.2f})")
    ax[n].imshow(s.mask, cmap="gray")
    ax[n].set_title("máscara de referência")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = Path("saidas/agml_bandas.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nFigura salva em {out}")


if __name__ == "__main__":
    main()
