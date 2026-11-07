"""Compara os treinos do aluno numa tabela e em gráficos (parte 5c).

Uso:
    python scripts/compare_runs.py                       # todos os treinos em runs/
    python scripts/compare_runs.py --runs lambda0 lambda1 lambda100

Lê, de cada pasta em runs/, o config.json e o historico.csv gravados pelo
scripts/train_student.py, e monta:

  - uma tabela com uma linha por treino: lambda, transformações, épocas
    treinadas, a época escolhida (a do melhor.pt, pelo critério do treino) e
    as métricas na validação nessa época, contra a referência e contra o
    professor. A primeira linha é o próprio professor TEBAKA na validação,
    a linha de base que o aluno precisa superar;
  - gráficos das curvas por época: F1 na validação (contra a referência e
    contra o professor), perda das máscaras e consistência.

Saídas em saidas/:
    comparacao.csv   a tabela
    comparacao.md    a mesma tabela em Markdown (para colar no relatório)
    comparacao.png   os gráficos

Treinos antigos, sem as colunas de validação (de antes da parte 5b),
entram com as métricas em branco: avalie-os com scripts/evaluate_student.py.

**Várias sementes (parte 7b).** Treinos com o mesmo nome e o sufixo _s<n>
(lambda100_s0, lambda100_s1, ..., como os que scripts/run_seeds.py cria)
viram uma linha só, chamada <nome>_s*, com a média e o desvio-padrão de cada
métrica entre as sementes ("0,52 ± 0,02") e o número de sementes na coluna
n. O desvio é o amostral (divide por n - 1). Um treino sem o sufixo nunca
entra no grupo, mesmo com o mesmo nome (lambda1 e lambda1_s0 são treinos
diferentes). Com --sem-agrupar, cada semente fica na sua linha.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METRICS = ["f1", "precisao", "revocacao", "iou_copa", "erro_contagem",
           "area_mae", "area_kl"]


def read_run(run_dir: Path) -> dict | None:
    """Config e histórico de um treino; None se a pasta não for um treino."""
    hist = run_dir / "historico.csv"
    cfg = run_dir / "config.json"
    if not hist.exists() or not cfg.exists():
        return None
    with open(hist, encoding="utf-8") as f:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None
    return {"nome": run_dir.name, "config": json.loads(cfg.read_text(encoding="utf-8")),
            "historico": rows}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def chosen_epoch(run: dict) -> dict:
    """A linha do histórico da época escolhida: a última marcada como
    'melhor' (a do melhor.pt); sem essa coluna, a última época."""
    rows = run["historico"]
    best = [r for r in rows if r.get("melhor") == 1.0]
    return best[-1] if best else rows[-1]


def summarize(run: dict) -> dict:
    cfg, row = run["config"], chosen_epoch(run)
    lam = cfg.get("peso_consistencia", 0)
    cons = cfg.get("consistencia", "") if lam else "desligada"
    out = {"treino": run["nome"], "etapa": cfg.get("etapa", 1), "lambda": lam,
           "consistencia": cons, "mu_max": cfg.get("mu_max", 0),
           "peso_kd": cfg.get("peso_kd", 0) if cfg.get("professor") else 0,
           "criterio": cfg.get("criterio", "nenhum"),
           "epocas": len(run["historico"]), "epoca_escolhida": int(row["epoca"]),
           "tempo_min": round(sum(r.get("tempo_s", 0) for r in run["historico"]) / 60)}
    for m in METRICS:
        out[f"ref_{m}"] = row.get(f"val_{m}", "")
        out[f"prof_{m}"] = row.get(f"valprof_{m}", "")
    return out


def teacher_baseline(dataset_dir, pseudo_dir) -> dict | None:
    """O professor TEBAKA contra a referência na validação (linha de base)."""
    try:
        from rsskd.data import AgmlTreeCrowns, mask_to_instances
        from rsskd.evaluation import CrownMetrics
        from rsskd.pseudolabels import PseudoLabelStore

        ds = AgmlTreeCrowns(dataset_dir, split="val")
        store = PseudoLabelStore(pseudo_dir)
        m = CrownMetrics()
        for i in range(len(ds)):
            m.add(store.load("val", i), mask_to_instances(ds[i].mask), index=i)
        s = m.summary()
    except (FileNotFoundError, OSError):
        return None
    out = {"treino": "professor TEBAKA", "etapa": "", "lambda": "", "consistencia": "",
           "mu_max": "", "peso_kd": "",
           "criterio": "", "epocas": "", "epoca_escolhida": "", "tempo_min": ""}
    out |= {f"ref_{k}": s[k] for k in METRICS}
    out |= {f"prof_{k}": "" for k in METRICS}
    return out


SEED_SUFFIX = re.compile(r"^(.*)_s(\d+)$")


def group_seeds(rows: list[dict]) -> list[dict]:
    """Junta as linhas dos treinos <nome>_s<n> numa linha <nome>_s* por grupo.

    Nas colunas numéricas, a linha do grupo traz a média, e a coluna
    <coluna>_dp traz o desvio-padrão amostral (0 com uma semente só). A
    época escolhida e o tempo também viram médias. Treinos sem o sufixo
    passam como estão, com n = 1, e nunca se misturam a um grupo de
    sementes de mesmo nome.
    """
    groups: dict[tuple[str, bool], list[dict]] = {}
    order: list[tuple[str, bool]] = []
    for r in rows:
        m = SEED_SUFFIX.match(str(r["treino"]))
        key = (m.group(1), True) if m else (str(r["treino"]), False)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    out = []
    for key in order:
        name, seeded = key
        members = groups[key]
        if not seeded:
            out.extend(m | {"n": 1} for m in members)
            continue
        row: dict = {"treino": f"{name}_s*", "n": len(members)}
        for col in members[0]:
            if col == "treino":
                continue
            vals = [m.get(col) for m in members]
            if all(isinstance(v, (int, float)) for v in vals):
                nums = [float(v) for v in vals if v is not None]
                row[col] = statistics.mean(nums)
                row[f"{col}_dp"] = statistics.stdev(nums) if len(nums) > 1 else 0.0
            else:
                row[col] = vals[0] if len(set(map(str, vals))) == 1 else "vários"
        out.append(row)
    return out


def to_markdown(rows: list[dict]) -> str:
    """Tabela em Markdown. As métricas vão com 2 casas (erro de contagem, 1)."""
    cols = [("treino", "Treino", None), ("n", "n", "g"), ("etapa", "Etapa", "g"),
            ("lambda", "λ", "g"),
            ("mu_max", "μ", "g"), ("peso_kd", "β", "g"),
            ("epoca_escolhida", "Época", "g"), ("ref_precisao", "P", ".2f"),
            ("ref_revocacao", "R", ".2f"), ("ref_f1", "F1", ".2f"),
            ("ref_iou_copa", "IoU copa", ".2f"),
            ("ref_erro_contagem", "Erro contagem", ".1f"),
            ("ref_area_mae", "MAE área (px)", ".0f"), ("ref_area_kl", "KL área", ".2f"),
            ("prof_f1", "F1 x professor", ".2f")]

    def cell(r, k, spec):
        v = r.get(k, "")
        if v == "" or v is None:
            return "—"
        if not (spec and isinstance(v, (int, float))):
            return str(v)
        dp = r.get(f"{k}_dp")
        if k.startswith(("ref_", "prof_")) and isinstance(dp, float) and r.get("n", 1) > 1:
            return f"{format(v, spec)} ± {format(dp, spec)}"
        return format(v, spec)

    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(cell(r, k, spec) for k, _, spec in cols) + " |")
    return "\n".join(lines) + "\n"


def plot(runs: list[dict], path: Path, teacher_f1: float | None):
    panels = [("val_f1", "F1 na validação x referência"),
              ("valprof_f1", "F1 na validação x professor"),
              ("loss_mask", "perda das máscaras (treino)"),
              ("loss_consistencia", "consistência (treino)")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, (key, title) in zip(ax.flat, panels):
        for run in runs:
            pts = [(r["epoca"], r[key]) for r in run["historico"]
                   if isinstance(r.get(key), float)]
            if pts:
                a.plot(*zip(*pts), marker=".", label=run["nome"])
        if key == "val_f1" and teacher_f1 is not None:
            a.axhline(teacher_f1, color="gray", linestyle="--", label="professor")
        a.set_title(title, fontsize=10)
        a.set_xlabel("época")
        if a.lines:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta-runs", default="runs")
    ap.add_argument("--runs", nargs="*", help="nomes dos treinos (padrão: todos)")
    ap.add_argument("--pasta", default="data/agml_tree_crown",
                    help="conjunto AgML, para a linha do professor")
    ap.add_argument("--copas", default="data/pseudo/agml")
    ap.add_argument("--saida", default="saidas")
    ap.add_argument("--sem-agrupar", action="store_true",
                    help="não junta as sementes (_s0, _s1, ...) numa linha só")
    args = ap.parse_args(argv)

    root = Path(args.pasta_runs)
    dirs = ([root / n for n in args.runs] if args.runs
            else sorted(p for p in root.iterdir() if p.is_dir()))
    runs = [r for r in (read_run(d) for d in dirs) if r is not None]
    if not runs:
        raise SystemExit(f"Nenhum treino encontrado em {root}/")

    rows = [summarize(r) for r in runs]
    if not args.sem_agrupar:
        rows = group_seeds(rows)
    teacher = teacher_baseline(args.pasta, args.copas)
    if teacher:
        rows.insert(0, teacher)

    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparacao.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    md = to_markdown(rows)
    (out / "comparacao.md").write_text(md, encoding="utf-8")
    plot(runs, out / "comparacao.png", teacher["ref_f1"] if teacher else None)
    print(md)
    print(f"Tabela e gráficos em {out}/comparacao.*")
    return rows


if __name__ == "__main__":
    main()
