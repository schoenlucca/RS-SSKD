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

**Incerteza (rsskd/stats.py).** Com dezenas de imagens, uma diferença de
0,01 no F1 pode vir de poucas imagens. Por isso a tabela traz, para o F1, o
intervalo de confiança de 95% por bootstrap sobre as imagens, e uma segunda
tabela compara cada método com o primeiro da lista (a linha de base, ex.: o
TEBAKA) por bootstrap **pareado**: a diferença de F1, o intervalo dela e o
valor p. Se o intervalo da diferença não cruza o zero, a vantagem se mantém
quando as imagens mudam. --base escolhe outra linha de base; --bootstrap 0
desliga.

**Área nas copas em comum.** O MAE de área de cada método só conta as copas
que ele acertou, e métodos diferentes acertam copas diferentes. A terceira
tabela mede o erro relativo de área de todos nas mesmas copas: as da
referência que todos acertaram.

Saídas em saidas/: previsoes_<divisão>.md (as três tabelas), .csv (resumo
por método) e _por_imagem.csv (contagens por imagem, para refazer o
bootstrap com outras combinações sem avaliar de novo).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.evaluation import CrownMetrics, common_area_error, format_summary  # noqa: E402
from rsskd.paths import HINT_FOLDER, require  # noqa: E402
from rsskd.predictions import PredictionStore, check_against  # noqa: E402
from rsskd.stats import bootstrap_ci, format_ci, paired_bootstrap  # noqa: E402

COLS = [("precisao", "P", ".2f"), ("revocacao", "R", ".2f"), ("f1", "F1", ".2f"),
        ("iou_copa", "IoU copa", ".2f"), ("iou_pixel", "IoU pixel", ".2f"),
        ("erro_contagem", "Erro contagem", ".1f"), ("area_mae", "MAE área (px)", ".0f"),
        ("area_erro_rel", "Erro rel. área", ".2f"), ("area_kl", "KL área", ".2f")]


def evaluate_folder(store, split, refs) -> CrownMetrics:
    """Mede uma pasta contra as copas da referência; devolve o CrownMetrics
    (o resumo sai de .summary(), as contagens por imagem de .rows)."""
    m = CrownMetrics()
    for i, ref in enumerate(refs):
        m.add(store.load(split, i), ref, index=i)
    return m


def uncertainty_tables(metrics: dict[str, CrownMetrics], base: int = 0,
                       n_boot: int = 10000) -> tuple[list[str], dict]:
    """Linhas em Markdown da comparação pareada e da área em comum, e os
    números delas (para o CSV e para os testes)."""
    names = list(metrics)
    lines, out = [], {"pareado": {}, "area_comum": common_area_error(metrics)}
    if n_boot > 0 and len(names) > 1:
        b = names[base]
        lines += ["", f"Diferença de F1 para {b}, bootstrap pareado nas mesmas imagens "
                      f"({n_boot} sorteios):", "",
                  "| Método | Diferença de F1 [IC 95%] | p |", "|---|---|---|"]
        for n in names:
            if n == b:
                continue
            d = paired_bootstrap(metrics[n].rows, metrics[b].rows, "f1", n_boot)
            out["pareado"][n] = d
            lines.append(f"| {n} | {format_ci(d)} | {d['p']:.3f} |")
    c = out["area_comum"]
    if len(names) > 1:
        lines += ["", f"Erro relativo de área nas {c['n_copas']} copas da referência "
                      "que todos os métodos acertaram:", "",
                  "| Método | Erro rel. área |", "|---|---|"]
        lines += [f"| {n} | {c[n]:.3f} |" if c["n_copas"] else f"| {n} | — |" for n in names]
    return lines, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pastas", nargs="+", help="pastas de previsões; a primeira é a linha de base")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pasta", default=str(DEFAULT_DIR), help="conjunto AgML")
    ap.add_argument("--saida", default="saidas")
    ap.add_argument("--base", type=int, default=0,
                    help="posição da linha de base na lista de pastas (padrão: a primeira)")
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="sorteios do bootstrap; 0 desliga os intervalos")
    args = ap.parse_args(argv)

    ds = AgmlTreeCrowns(args.pasta, split=args.split)
    refs = [mask_to_instances(ds[i].mask) for i in range(len(ds))]
    rows, metrics = [], {}
    for folder in args.pastas:
        require(folder, "a pasta de previsões", HINT_FOLDER)
        store = PredictionStore(folder)
        problems = check_against(store, args.split, ds)
        if problems:
            raise SystemExit(f"{folder}: " + "; ".join(problems))
        m = evaluate_folder(store, args.split, refs)
        s = m.summary()
        name = store.info().get("metodo") or Path(folder).name
        if name in metrics:
            name = f"{name} ({folder})"
        metrics[name] = m
        if args.bootstrap > 0:
            ci = bootstrap_ci(m.rows, "f1", args.bootstrap)
            s |= {"f1_inf": ci["inf"], "f1_sup": ci["sup"]}
        print(f"{name}: {format_summary(s)}")
        rows.append({"previsoes": name, "pasta": folder} | s)

    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"previsoes_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    with open(out / f"previsoes_{args.split}_por_imagem.csv", "w", newline="",
              encoding="utf-8") as f:
        per_image = [{"previsoes": n} | r for n, m in metrics.items() for r in m.rows]
        w = csv.DictWriter(f, fieldnames=list(per_image[0]))
        w.writeheader()
        w.writerows(per_image)

    ci_col = args.bootstrap > 0
    head = [h for _, h, _ in COLS] + (["F1 [IC 95%]"] if ci_col else [])
    lines = [f"Avaliação em {args.split} ({len(ds)} imagens), contra a referência do AgML.", "",
             "| Previsões | " + " | ".join(head) + " |",
             "|" + "---|" * (len(head) + 1)]
    for r in rows:
        cells = [format(r[k], spec) for k, _, spec in COLS]
        if ci_col:
            cells.append(format_ci({"valor": r["f1"], "inf": r["f1_inf"], "sup": r["f1_sup"]}))
        lines.append(f"| {r['previsoes']} | " + " | ".join(cells) + " |")
    extra, _ = uncertainty_tables(metrics, args.base, args.bootstrap)
    md = "\n".join(lines + extra) + "\n"
    (out / f"previsoes_{args.split}.md").write_text(md, encoding="utf-8")
    print("\n".join(extra))
    print(f"\nTabelas em {out}/previsoes_{args.split}.md")
    return rows


if __name__ == "__main__":
    main()
