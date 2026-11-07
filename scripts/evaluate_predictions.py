"""Avalia pastas de previsões com as mesmas métricas (parte 7c).

Uso:
    python scripts/evaluate_predictions.py previsoes/felipe_sam previsoes/etapa2 --split test
    python scripts/evaluate_predictions.py data/pseudo/agml previsoes/etapa2 --split val

Cada pasta segue o formato de rsskd/predictions.py (uma imagem de etiquetas
por recorte). Todas são medidas pelo mesmo rsskd.evaluation.CrownMetrics,
contra a mesma referência do AgML: precisão, revocação e F1 por copa
(IoU >= 0,5), IoU, erro de contagem e as métricas de área de Harmon. A pasta
das copas do professor TEBAKA (data/pseudo/agml) tem o mesmo formato, então
o professor entra na mesma tabela.

Antes de medir, cada pasta é conferida: imagens faltando, tamanho errado ou
ordem trocada (pela soma de verificação do info.json) interrompem a
avaliação, porque comparar a previsão de uma imagem com a referência de
outra daria números sem sentido, sem nenhum erro aparente.

Saídas em saidas/: previsoes_<divisão>.md (tabela) e .csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.evaluation import CrownMetrics, format_summary  # noqa: E402
from rsskd.paths import HINT_FOLDER, require  # noqa: E402
from rsskd.predictions import PredictionStore, check_against  # noqa: E402

COLS = [("precisao", "P", ".2f"), ("revocacao", "R", ".2f"), ("f1", "F1", ".2f"),
        ("iou_copa", "IoU copa", ".2f"), ("iou_pixel", "IoU pixel", ".2f"),
        ("erro_contagem", "Erro contagem", ".1f"), ("area_mae", "MAE área (px)", ".0f"),
        ("area_kl", "KL área", ".2f")]


def evaluate_folder(store, split, dataset, refs) -> dict:
    m = CrownMetrics()
    for i, ref in enumerate(refs):
        m.add(store.load(split, i), ref, index=i)
    return m.summary()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pastas", nargs="+", help="pastas de previsões")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pasta", default=str(DEFAULT_DIR), help="conjunto AgML")
    ap.add_argument("--saida", default="saidas")
    args = ap.parse_args(argv)

    ds = AgmlTreeCrowns(args.pasta, split=args.split)
    refs = [mask_to_instances(ds[i].mask) for i in range(len(ds))]
    rows = []
    for folder in args.pastas:
        require(folder, "a pasta de previsões", HINT_FOLDER)
        store = PredictionStore(folder)
        problems = check_against(store, args.split, ds)
        if problems:
            raise SystemExit(f"{folder}: " + "; ".join(problems))
        s = evaluate_folder(store, args.split, ds, refs)
        name = store.info().get("metodo") or Path(folder).name
        print(f"{name}: {format_summary(s)}")
        rows.append({"previsoes": name, "pasta": folder} | s)

    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"previsoes_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Avaliação em {args.split} ({len(ds)} imagens), contra a referência do AgML.", "",
             "| Previsões | " + " | ".join(h for _, h, _ in COLS) + " |",
             "|" + "---|" * (len(COLS) + 1)]
    for r in rows:
        lines.append(f"| {r['previsoes']} | "
                     + " | ".join(format(r[k], spec) for k, _, spec in COLS) + " |")
    md = "\n".join(lines) + "\n"
    (out / f"previsoes_{args.split}.md").write_text(md, encoding="utf-8")
    print(f"\nTabela em {out}/previsoes_{args.split}.md")
    return rows


if __name__ == "__main__":
    main()
