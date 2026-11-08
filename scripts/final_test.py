"""Avaliação final no teste, num passo só, para o relatório.

Uso:
    python scripts/final_test.py "Etapa 1=runs/curto_l0_s*" "Etapa 2=runs/e2_kd_s*"
    python scripts/final_test.py "Etapa 2=runs/e2_kd_longo_s*" --extra SAM=previsoes/felipe_sam

Cada argumento é "Nome=padrão", com o padrão casando as pastas das sementes
de um método (runs/e2_kd_s0, _s1, _s2). Para cada semente, o script:

  1. exporta as previsões do modelo escolhido na divisão pedida (padrão:
     melhor.pt no teste) para previsoes/<treino>/ (pula se já estiverem lá,
     do mesmo modelo);
  2. mede contra a referência, com o mesmo CrownMetrics do professor.

E monta, em saidas/teste_final.md:

  - uma linha por método: média ± desvio entre as sementes de P, R, F1,
    erro de contagem, erro relativo de área e KL de área; o TEBAKA
    (data/pseudo/agml) é sempre a primeira linha, e --extra acrescenta
    pastas prontas, como o SAM do Felipe;
  - a diferença de F1 de cada método para o TEBAKA, por bootstrap pareado
    nas imagens, com as sementes juntas (as contagens de cada imagem somadas
    entre as sementes, o que equivale ao F1 médio da imagem);
  - o erro relativo de área nas copas que todos os métodos acertaram (com a
    1ª semente de cada método).

**O teste é para o fim.** Rode-o uma vez, com as versões escolhidas pela
validação. Cada execução no teste fica anotada em saidas/registro_teste.csv
(data e o que foi avaliado), para o relatório poder dizer, com honestidade,
quantas vezes o teste foi olhado. --split val faz o mesmo na validação, sem
anotar.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_predictions import evaluate_folder  # noqa: E402
from export_predictions import main as export  # noqa: E402

from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.evaluation import CrownMetrics, common_area_error  # noqa: E402
from rsskd.paths import MissingFileError, require  # noqa: E402
from rsskd.predictions import PredictionStore, check_against  # noqa: E402
from rsskd.stats import COLUMNS, format_ci, paired_bootstrap  # noqa: E402

METRICS = [("precisao", "P", ".2f"), ("revocacao", "R", ".2f"), ("f1", "F1", ".3f"),
           ("erro_contagem", "Erro contagem", ".1f"), ("area_erro_rel", "Erro rel. área", ".2f"),
           ("area_kl", "KL área", ".2f")]


def parse_groups(items: list[str]) -> list[tuple[str, list[Path]]]:
    """"Nome=padrão" -> (nome, pastas das sementes em ordem)."""
    groups = []
    for it in items:
        name, sep, pattern = it.partition("=")
        if not sep:
            name, pattern = Path(it).name, it
        runs = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
        if not runs:
            raise MissingFileError(f"Nenhum treino casa com {pattern!r} (método {name}).")
        groups.append((name, runs))
    return groups


def predictions_for(run: Path, model: str, split: str, args) -> Path:
    """Pasta com as previsões do modelo na divisão; exporta se faltar."""
    folder = Path(args.previsoes) / run.name
    store = PredictionStore(folder)
    done = store.info().get("modelo") == model and split in store.info().get("imagens", {})
    if not done:
        export(["--run", str(run), "--modelo", model, "--divisoes", split,
                "--saida", str(folder), "--dispositivo", args.dispositivo,
                "--limiar-score", str(args.limiar_score)])
        info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
        info["modelo"] = model
        (folder / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return folder


def pooled_rows(metrics: list[CrownMetrics]) -> list[dict]:
    """Contagens por imagem somadas entre as sementes de um método."""
    out = []
    for per_seed in zip(*(m.rows for m in metrics)):
        row = {"indice": per_seed[0]["indice"]}
        row |= {c: sum(float(r[c]) for r in per_seed) for c in COLUMNS}
        out.append(row)
    return out


def mean_sd(values: list[float], spec: str) -> str:
    m = format(statistics.mean(values), spec)
    if len(values) < 2:
        return m
    return f"{m} ± {format(statistics.stdev(values), spec)}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("metodos", nargs="+", help='"Nome=padrão das pastas", ex.: "Etapa 2=runs/e2_kd_s*"')
    ap.add_argument("--extra", nargs="*", default=[],
                    help='pastas de previsões prontas, "Nome=pasta" (ex.: SAM=previsoes/felipe_sam)')
    ap.add_argument("--modelo", default="melhor")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--pasta", default=str(DEFAULT_DIR), help="conjunto AgML")
    ap.add_argument("--copas", default="data/pseudo/agml", help="copas do TEBAKA")
    ap.add_argument("--previsoes", default="previsoes")
    ap.add_argument("--saida", default="saidas")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--limiar-score", type=float, default=0.5)
    ap.add_argument("--dispositivo", default="auto")
    args = ap.parse_args(argv)

    ds = AgmlTreeCrowns(args.pasta, split=args.split)
    refs = [mask_to_instances(ds[i].mask) for i in range(len(ds))]

    # Cada método: lista de CrownMetrics, um por semente.
    methods: dict[str, list[CrownMetrics]] = {}
    sources: dict[str, list[str]] = {}
    folders = [("TEBAKA", [Path(args.copas)])]
    for it in args.extra:
        name, _, folder = it.partition("=")
        folders.append((name, [Path(folder or name)]))
    for name, runs in parse_groups(args.metodos):
        folders.append((name, [predictions_for(r, args.modelo, args.split, args)
                               for r in runs]))
    for name, paths in folders:
        methods[name], sources[name] = [], []
        for folder in paths:
            require(folder, f"a pasta de previsões de {name}")
            store = PredictionStore(folder)
            problems = check_against(store, args.split, ds)
            if problems:
                raise SystemExit(f"{folder}: " + "; ".join(problems))
            methods[name].append(evaluate_folder(store, args.split, refs))
            sources[name].append(str(folder))

    summaries = {n: [m.summary() for m in ms] for n, ms in methods.items()}
    head = ["Método", "n"] + [h for _, h, _ in METRICS]
    lines = [f"Avaliação final em {args.split} ({len(ds)} imagens, modelo {args.modelo}.pt), "
             "contra a referência do AgML. Média ± desvio entre sementes.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n, ss in summaries.items():
        cells = [mean_sd([s[k] for s in ss], spec) for k, _, spec in METRICS]
        lines.append(f"| {n} | {len(ss)} | " + " | ".join(cells) + " |")

    pooled = {n: pooled_rows(ms) for n, ms in methods.items()}
    comparisons = {}
    if args.bootstrap > 0:
        lines += ["", "Diferença de F1 para o TEBAKA, bootstrap pareado nas imagens "
                      f"(sementes juntas, {args.bootstrap} sorteios):", "",
                  "| Método | Diferença de F1 [IC 95%] | p |", "|---|---|---|"]
        for n in methods:
            if n == "TEBAKA":
                continue
            d = paired_bootstrap(pooled[n], pooled["TEBAKA"], "f1", args.bootstrap)
            comparisons[n] = d
            lines.append(f"| {n} | {format_ci(d)} | {d['p']:.3f} |")
    common = common_area_error({n: ms[0] for n, ms in methods.items()})
    lines += ["", f"Erro relativo de área nas {common['n_copas']} copas da referência que "
                  "todos os métodos acertaram (1ª semente de cada):", "",
              "| Método | Erro rel. área |", "|---|---|"]
    lines += [f"| {n} | {common[n]:.3f} |" if common["n_copas"] else f"| {n} | — |"
              for n in methods]

    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    tag = "teste_final" if args.split == "test" else "final_val"
    md = "\n".join(lines) + "\n"
    (out / f"{tag}.md").write_text(md, encoding="utf-8")
    (out / f"{tag}.json").write_text(json.dumps(
        {"split": args.split, "modelo": args.modelo, "fontes": sources,
         "resumos": summaries, "diferenca_f1_tebaka": comparisons, "area_comum": common},
        indent=2, ensure_ascii=False), encoding="utf-8")
    if args.split == "test":
        log = out / "registro_teste.csv"
        before = sum(1 for _ in open(log, encoding="utf-8")) - 1 if log.exists() else 0
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not before and not log.stat().st_size:
                w.writerow(["data", "modelo", "metodos"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), args.modelo,
                        "; ".join(f"{n}: {', '.join(s)}" for n, s in sources.items())])
        if before:
            print(f"Atenção: esta é a avaliação no teste nº {before + 1} "
                  f"(veja {log}).")
    print(md)
    print(f"Tabelas em {out}/{tag}.md")
    return summaries, comparisons


if __name__ == "__main__":
    main()
