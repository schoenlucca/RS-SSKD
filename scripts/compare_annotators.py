"""Concordância entre duas pessoas que rotularam as mesmas imagens.

Uso:
    python3 scripts/compare_annotators.py rotulagem/lucca.json rotulagem/felipe.json \\
        --nomes Lucca Felipe

Num cerradão, até a olho é difícil dizer onde termina uma copa e começa a
outra. Antes de usar a referência para medir os métodos, vale saber quanto
duas pessoas concordam entre si: é o teto realista do F1. Um método que
chega perto da concordância entre as pessoas está no limite do que a
referência consegue medir.

O ideal é as duas pessoas rotularem, de forma independente, um pedaço das
imagens (umas 10). O script mede, nas imagens que estão nos dois arquivos e
na escala do modelo (a mesma da avaliação):
  - o F1 por copa de uma pessoa contra a outra (IoU >= 0,5; é simétrico);
  - o IoU médio das copas casadas (o quanto os contornos coincidem);
  - o número de copas de cada uma.

Saídas em saidas/: concordancia_rotulos.md e .csv (uma linha por imagem).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.coco import downscale_labels, load_coco  # noqa: E402
from rsskd.data import match_instances  # noqa: E402
from rsskd.evaluation import CrownMetrics  # noqa: E402
from rsskd.labeling import MANIFEST, load_manifest  # noqa: E402
from rsskd.paths import require  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("coco", nargs=2, help="os dois arquivos COCO")
    ap.add_argument("--nomes", nargs=2, default=["A", "B"])
    ap.add_argument("--manifesto", default=f"rotulagem/{MANIFEST}")
    ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--saida", default="saidas")
    args = ap.parse_args(argv)

    _, crops = load_manifest(require(args.manifesto, "o manifesto dos recortes"))
    size, factor = crops[0].tamanho, round(1 / crops[0].escala)
    a_all, b_all = (load_coco(require(p, "o arquivo COCO"), size) for p in args.coco)
    common = [c.arquivo for c in crops if c.arquivo in a_all and c.arquivo in b_all]
    if not common:
        raise SystemExit("Nenhuma imagem rotulada pelas duas pessoas.")

    m = CrownMetrics()
    rows = []
    for i, name in enumerate(common):
        a = downscale_labels(a_all[name], factor, args.min_area)
        b = downscale_labels(b_all[name], factor, args.min_area)
        m.add(a, b, index=i)
        r = match_instances(a, b)
        rows.append({"imagem": name, f"copas_{args.nomes[0]}": int(a.max()),
                     f"copas_{args.nomes[1]}": int(b.max()), "f1": round(float(r["f1"]), 3)})
    s = m.summary()
    na, nb = args.nomes
    lines = [f"Concordância entre {na} e {nb} em {len(common)} imagens rotuladas pelos dois "
             "(escala do modelo, IoU >= 0,5).", "",
             "| F1 por copa | IoU médio das copas casadas | Copas " + na + " | Copas " + nb + " |",
             "|---|---|---|---|",
             f"| {s['f1']:.3f} | {s['iou_copa']:.3f} | "
             f"{sum(r[f'copas_{na}'] for r in rows)} | {sum(r[f'copas_{nb}'] for r in rows)} |"]
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    (out / "concordancia_rotulos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "concordancia_rotulos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    return s, rows


if __name__ == "__main__":
    main()
