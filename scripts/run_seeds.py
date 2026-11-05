"""Roda o mesmo treino com várias sementes (parte 7b).

Uso:
    python scripts/run_seeds.py --nome lambda100 --sementes 0 1 2 -- --peso-consistencia 100
    python scripts/run_seeds.py --nome etapa2 --sementes 0 1 2 -- --professor runs/lambda100_s0/melhor.pt

Tudo o que vem depois de "--" vai igual para scripts/train_student.py. Cada
semente vira um treino <nome>_s<semente> (lambda100_s0, lambda100_s1, ...),
e o scripts/compare_runs.py junta esses treinos numa linha só, com média e
desvio-padrão.

**Por que várias sementes.** A semente fixa tudo o que é sorteado no treino:
os pesos iniciais das cabeças do Mask R-CNN, a ordem das imagens, as
rotações sorteadas. Dois treinos iguais com sementes diferentes dão F1
diferentes, às vezes por alguns centésimos. Sem repetir, não dá para saber
se uma diferença de 0,02 entre duas versões é do método ou do sorteio. Três
sementes é o mínimo usual; o desvio-padrão diz quanto o resultado varia.

Treinos que já terminaram (com melhor.pt e a última época no histórico) são
pulados; um treino interrompido é retomado (--continuar).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_student import main as train  # noqa: E402
from train_student import parse_args as train_args  # noqa: E402


def finished(run_dir: Path, epochs: int) -> bool:
    """O treino já terminou? (melhor.pt existe e o histórico chegou ao fim,
    ou parou antes por parada antecipada, o que o config não diz; na
    dúvida, o --continuar do train_student retoma sem refazer.)"""
    hist = run_dir / "historico.csv"
    if not (run_dir / "melhor.pt").exists() or not hist.exists():
        return False
    lines = hist.read_text(encoding="utf-8").strip().splitlines()
    return len(lines) - 1 >= epochs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)

    runs = []
    for seed in args.sementes:
        name = f"{args.nome}_s{seed}"
        cmd = rest + ["--nome", name, "--semente", str(seed)]
        opts = train_args(cmd)
        run_dir = Path(opts.saida) / name
        if finished(run_dir, opts.epocas):
            print(f"{name}: já terminado, pulando")
        else:
            print(f"\n=== {name} ===")
            train(cmd + (["--continuar"] if run_dir.exists() else []))
        runs.append(run_dir)
    print(f"\nPronto: {[r.name for r in runs]}. Compare com: python scripts/compare_runs.py")
    return runs


if __name__ == "__main__":
    main()
