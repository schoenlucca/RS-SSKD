"""Calcula a normalização da entrada do aluno (média e desvio por canal).

Uso:
    python scripts/compute_input_stats.py

Lê as bandas do NIR e do vermelho em configs/agml_teacher.json (seção
"bandas"), passa por todas as imagens de **treino** do conjunto AgML e grava
configs/agml_entrada.json. Esse arquivo vai para o git: é pequeno e
garante que todo treino e toda avaliação usem a mesma normalização.

Só o treino entra na conta. Validação e teste são normalizados com os
mesmos números, como se fossem imagens novas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns  # noqa: E402
from rsskd.inputs import compute_input_spec  # noqa: E402
from rsskd.teacher import load_config  # noqa: E402

DEFAULT_OUT = Path("configs/agml_entrada.json")


def channel_names(bands_cfg: dict, n_bands: int) -> list[str]:
    """Nomes das bandas pelo índice (ex.: 0 -> "vermelho"), mais "ndvi"."""
    by_index = {v: k for k, v in bands_cfg.items() if not k.startswith("_")}
    return [by_index.get(i, f"banda_{i}") for i in range(n_bands)] + ["ndvi"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    ap.add_argument("--config", default="configs/agml_teacher.json")
    ap.add_argument("--saida", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    bands_cfg = load_config(args.config)["bandas"]
    ds = AgmlTreeCrowns(args.pasta, split="train")
    n_bands = ds[0].bands.shape[0]
    names = channel_names(bands_cfg, n_bands)

    spec = compute_input_spec((ds[i].bands for i in range(len(ds))),
                              bands_cfg["nir"], bands_cfg["vermelho"], names)
    spec.save(args.saida, comment=(
        f"Normalização da entrada do aluno: média e desvio de cada canal nas "
        f"{len(ds)} imagens de treino de {args.pasta}. Gerado por "
        f"scripts/compute_input_stats.py."))

    print(f"{len(ds)} imagens de treino, {spec.in_channels} canais")
    for name, m, s in zip(spec.channels, spec.mean, spec.std):
        print(f"  {name:>10}: média {m:9.3f}   desvio {s:8.3f}")
    print(f"Gravado em {args.saida}")
    return spec


if __name__ == "__main__":
    main()
