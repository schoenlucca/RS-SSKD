"""Tabelas do artigo em LaTeX, direto dos resultados (sem copiar à mão).

Uso:
    python scripts/latex_tables.py saidas/teste_final_tta.json
    python scripts/latex_tables.py saidas/teste_final_tta.json --metricas f1 f1_iou75 area_kl \\
        --legenda "Resultados no teste do AgML" --rotulo tab:agml --decimal virgula

Lê o .json gravado por scripts/final_test.py e escreve, em saidas/tabelas/
(ou --saida), um arquivo .tex com uma tabela no estilo booktabs:

  - uma linha por método, na ordem do json (o TEBAKA primeiro);
  - cada métrica como média ± desvio entre as sementes (só a média quando
    há uma semente);
  - o melhor valor de cada coluna em negrito (o maior, ou o menor nas
    métricas de erro);
  - uma última coluna com a diferença de F1 para o TEBAKA e o intervalo de
    95% do bootstrap pareado, marcada com * quando o intervalo não cruza o
    zero.

No preâmbulo do artigo: \\usepackage{booktabs}. A tabela entra com
\\input{tabelas/teste_final_tta.tex}.

--decimal virgula escreve 0,535 (artigo em português); o padrão é o ponto
(artigo em inglês, como no IEEE).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Nome na tabela e se o menor valor é o melhor. O erro de contagem (copas
# por imagem) sai com uma casa; as outras, com --casas.
METRICS = {
    "f1": ("F1", False), "f1_iou25": ("F1$_{0.25}$", False),
    "f1_iou75": ("F1$_{0.75}$", False), "precisao": ("P", False), "revocacao": ("R", False),
    "iou_copa": ("IoU$_{copa}$", False), "iou_pixel": ("IoU$_{pixel}$", False),
    "erro_contagem": ("Err. cont.", True), "area_erro_rel": ("Err. área", True),
    "area_kl": ("KL área", True),
}
DEFAULT = ["f1", "f1_iou25", "f1_iou75", "precisao", "revocacao", "iou_copa",
           "erro_contagem", "area_kl"]


def fmt(x: float, digits: int, decimal: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if decimal == "virgula" else s


def table(result: dict, metrics: list[str], caption: str, label: str,
          decimal: str = "ponto", digits: int = 3) -> str:
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Métricas desconhecidas: {unknown}. Use: {list(METRICS)}")
    summaries: dict[str, list[dict]] = result["resumos"]
    diffs: dict[str, dict] = result.get("diferenca_f1_tebaka", {})
    means = {n: {m: statistics.mean(s[m] for s in ss) for m in metrics if m in ss[0]}
             for n, ss in summaries.items()}
    best = {}
    for m in metrics:
        vals = [v[m] for v in means.values() if m in v]
        if vals:
            best[m] = min(vals) if METRICS[m][1] else max(vals)

    cols = "l" + "c" * len(metrics) + ("c" if diffs else "")
    head = ["Método"] + [METRICS[m][0] for m in metrics] + (["$\\Delta$F1 [IC 95\\%]"]
                                                           if diffs else [])
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    for name, ss in summaries.items():
        cells = [name.replace("_", "\\_")]
        for m in metrics:
            if m not in ss[0]:
                cells.append("--")
                continue
            vals = [s[m] for s in ss]
            mean = statistics.mean(vals)
            d = 1 if m == "erro_contagem" else digits
            txt = fmt(mean, d, decimal)
            if len(vals) > 1:
                txt += " $\\pm$ " + fmt(statistics.stdev(vals), d, decimal)
            if round(mean, d) == round(best[m], d):
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        if diffs:
            diff = diffs.get(name)
            if diff is None:
                cells.append("--")
            else:
                sig = "*" if diff["inf"] > 0 or diff["sup"] < 0 else ""
                sign = "+" if diff["diferenca"] > 0 else ""
                cells.append(f"{sign}{fmt(diff['diferenca'], digits, decimal)}{sig} "
                             f"[{fmt(diff['inf'], digits, decimal)}; "
                             f"{fmt(diff['sup'], digits, decimal)}]")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("resultados", nargs="+", help=".json do scripts/final_test.py")
    ap.add_argument("--metricas", nargs="+", default=DEFAULT, choices=list(METRICS))
    ap.add_argument("--legenda", default=None)
    ap.add_argument("--rotulo", default=None)
    ap.add_argument("--decimal", default="ponto", choices=["ponto", "virgula"])
    ap.add_argument("--casas", type=int, default=3)
    ap.add_argument("--saida", default="saidas/tabelas")
    args = ap.parse_args(argv)

    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for path in map(Path, args.resultados):
        result = json.loads(path.read_text(encoding="utf-8"))
        tta = " com TTA" if result.get("tta") else ""
        caption = args.legenda or (f"Resultados em {result['split']}{tta} "
                                   "(média $\\pm$ desvio entre sementes).")
        label = args.rotulo or f"tab:{path.stem}"
        tex = table(result, args.metricas, caption, label, args.decimal, args.casas)
        dest = out / f"{path.stem}.tex"
        dest.write_text(tex, encoding="utf-8")
        written.append(dest)
        print(f"{dest}")
    return written


if __name__ == "__main__":
    main()
