"""Ajuste visual do professor TEBAKA, sem referência (para o Cerrado).

Uso:
    python3 scripts/tune_teacher.py
    python3 scripts/tune_teacher.py --grade split_by=distance,nir min_distance=12,20 sigma=2,4
    python3 scripts/tune_teacher.py --grade ... --gravar 3      # grava a linha 3 na configuração

No AgML, o professor foi calibrado por busca em grade contra a referência da
validação (scripts/evaluate_teacher.py --grade). No Cerrado não há
referência, então a escolha é feita olhando. Este script:

  1. sorteia --n recortes do **treino** (nunca da validação ou do teste:
     ajustar olhando imagens de avaliação deixaria o resultado otimista);
  2. roda o professor com cada combinação da grade (o produto de todos os
     valores de cada parâmetro), partindo da configuração --config;
  3. monta uma figura com uma linha por combinação e uma coluna por recorte:
     a imagem em falsa cor (NIR, vermelho, verde: vegetação em vermelho) com
     os contornos das copas em amarelo e o número de copas;
  4. grava um resumo por combinação: copas por recorte, área mediana da copa
     e a fração da vegetação coberta por copas.

O que procurar na figura: um contorno por copa visível. Contornos dentro de
uma mesma copa = separação demais (aumentar min_distance ou sigma); copas
vizinhas num contorno só = separação de menos (diminuir); manchas de capim
ou sombra contornadas = área mínima ou limiar baixos.

Com --gravar N, os parâmetros da linha N (contada a partir de 0, como na
figura) vão para a seção "professor" da configuração. O ajuste visual deve
ser descrito no relatório como tal: escolha a olho, em recortes de treino.

Parâmetros da grade: os do TebakaTeacher (th1, th2, min_distance, min_area,
split_by, sigma, nlm_h...) e "th", que muda th1 e th2 juntos. Valores
numéricos ou textos ("otsu", "nir").

Saídas em saidas/: ajuste_professor.png e ajuste_professor.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import AgmlTreeCrowns  # noqa: E402
from rsskd.paths import require  # noqa: E402
from rsskd.teacher import TebakaTeacher, load_config  # noqa: E402

DEFAULT_GRID = ["split_by=distance,nir", "min_distance=12,20", "sigma=3"]


def parse_value(v: str):
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def parse_grid(items: list[str]) -> list[dict]:
    """["a=1,2", "b=x"] -> [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]."""
    keys, values = [], []
    for it in items:
        key, sep, vals = it.partition("=")
        if not sep or not vals:
            raise ValueError(f"Use parâmetro=valor1,valor2 (recebi {it!r}).")
        keys.append(key)
        values.append([parse_value(v) for v in vals.split(",")])
    combos = []
    for vals in itertools.product(*values):
        c = dict(zip(keys, vals))
        if "th" in c:
            c["th1"] = c["th2"] = c.pop("th")
        combos.append(c)
    return combos


def false_color(bands: np.ndarray, bands_cfg: dict) -> np.ndarray:
    """(H, W, 3) em [0, 1]: NIR, vermelho, verde, cada um esticado entre os
    percentis 2 e 98 (só para ver; o professor usa os valores originais)."""
    chans = [bands[bands_cfg[k]].astype(float) for k in ("nir", "vermelho", "verde")]
    out = []
    for c in chans:
        lo, hi = np.percentile(c, [2, 98])
        out.append(np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1))
    return np.dstack(out)


def summarize(results: list[dict]) -> dict:
    """Resumo de uma combinação nos recortes sorteados."""
    counts = [r["n_crowns"] for r in results]
    areas = [a for r in results for a in np.bincount(r["instances"].ravel())[1:] if a > 0]
    veg = sum(int(r["vegetation"].sum()) for r in results)
    covered = sum(int((r["instances"] > 0).sum()) for r in results)
    return {"copas_por_recorte": float(np.mean(counts)),
            "area_mediana_px": float(np.median(areas)) if areas else 0.0,
            "cobertura_da_vegetacao": covered / veg if veg else 0.0}


def save_choice(config_path: Path, params: dict) -> None:
    cfg = load_config(config_path)
    prof = cfg["professor"]
    prof.update(params)
    prof["_ajuste"] = ("Escolhido a olho por scripts/tune_teacher.py em recortes de treino, "
                       "sem referência: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="data/cerrado")
    ap.add_argument("--config", default="configs/cerrado_teacher.json")
    ap.add_argument("--grade", nargs="+", default=DEFAULT_GRID,
                    help='combinações, ex.: split_by=distance,nir min_distance=12,20')
    ap.add_argument("--n", type=int, default=6, help="recortes sorteados do treino")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", default="saidas")
    ap.add_argument("--gravar", type=int, default=None,
                    help="grava na configuração a combinação desta linha (a partir de 0)")
    args = ap.parse_args(argv)

    config = require(args.config, "a configuração do professor")
    combos = parse_grid(args.grade)
    if args.gravar is not None:
        if not 0 <= args.gravar < len(combos):
            raise SystemExit(f"--gravar deve estar entre 0 e {len(combos) - 1}.")
        replace(TebakaTeacher.from_config(config), **combos[args.gravar])   # confere os nomes
        save_choice(config, combos[args.gravar])
        print(f"Gravado em {config}: {combos[args.gravar]}")
        return combos[args.gravar]
    if len(combos) > 16:
        raise SystemExit(f"{len(combos)} combinações: a figura fica ilegível. Use até 16.")

    bands_cfg = load_config(config)["bandas"]
    base = TebakaTeacher.from_config(config)
    ds = AgmlTreeCrowns(args.pasta, split="train")
    rng = np.random.default_rng(args.semente)
    idx = sorted(rng.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    samples = [ds[i] for i in idx]
    images = [false_color(s.bands, bands_cfg) for s in samples]

    fig, axes = plt.subplots(len(combos), len(samples), squeeze=False,
                             figsize=(2.4 * len(samples), 2.6 * len(combos)))
    rows = []
    for r, params in enumerate(combos):
        teacher = replace(base, **params)
        results = [teacher(s.bands) for s in samples]
        summary = summarize(results)
        rows.append({"linha": r} | params | summary)
        for c, (img, res) in enumerate(zip(images, results)):
            ax = axes[r][c]
            shown = img.copy()
            shown[find_boundaries(res["instances"], mode="inner")] = (1.0, 1.0, 0.0)
            ax.imshow(shown)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{res['n_crowns']} copas", fontsize=8)
            if c == 0:
                label = "\n".join(f"{k}={v}" for k, v in params.items())
                ax.set_ylabel(f"[{r}]\n{label}", fontsize=7)
        print(f"[{r}] {params}: {summary['copas_por_recorte']:.1f} copas/recorte, "
              f"área mediana {summary['area_mediana_px']:.0f} px, "
              f"cobertura {summary['cobertura_da_vegetacao']:.0%}")
    fig.suptitle(f"Professor em {len(samples)} recortes de treino ({Path(args.pasta).name}); "
                 "copas em amarelo", fontsize=10)
    fig.tight_layout()
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ajuste_professor.png", dpi=110)
    plt.close(fig)
    with open(out / "ajuste_professor.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"Figura e resumo em {out}/ajuste_professor.*  "
          f"(recortes {idx}; escolha uma linha e grave com --gravar N)")
    return rows


if __name__ == "__main__":
    main()
