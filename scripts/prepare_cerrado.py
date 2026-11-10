"""Prepara as fotos do Cerrado (Mavic 3M) no formato do AgML.

Uso, no servidor do LISA:
    python3 scripts/prepare_cerrado.py \\
        --origem /mnt/sdb-seagate/graduacao/datasets/projeto_cerrado \\
        --treino 2025-01-27-voo1 2025-01-28 --val 2025-01-27-voo2

Junta as 4 bandas de cada disparo (verde, vermelho, Red-Edge e NIR), reduz
a foto pela metade, corta em recortes de 256×256 e grava, em data/cerrado/,
um arquivo Parquet por divisão (train, val, test) e o manifesto.json com o
que entrou. Detalhes e motivos em rsskd/cerrado.py.

A divisão é por voo: cada voo vai inteiro para uma divisão só (fotos do
mesmo voo se sobrepõem, e misturar treino e teste no mesmo voo deixaria o
resultado otimista). Os voos de seca (2024-08-28, 2025-06-16) ficam de fora
do padrão: na seca muitas árvores perdem folhas e o NDVI cai.

Depois disso, o resto do pipeline roda com --pasta data/cerrado e a
configuração do Cerrado (configs/cerrado_teacher.json), por exemplo:
    python3 scripts/generate_pseudolabels.py --pasta data/cerrado \\
        --config configs/cerrado_teacher.json --saida data/pseudo/cerrado
    python3 scripts/compute_input_stats.py --pasta data/cerrado \\
        --config configs/cerrado_teacher.json --saida configs/cerrado_entrada.json
Antes de gerar as copas, os parâmetros do professor precisam ser ajustados
para a mata fechada do Cerrado (os da configuração são só o ponto de partida).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.cerrado import build_dataset  # noqa: E402
from rsskd.paths import require  # noqa: E402

ORIGEM = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default=ORIGEM, help="pasta com uma subpasta por voo")
    ap.add_argument("--saida", default="data/cerrado")
    ap.add_argument("--treino", nargs="+", default=["2025-01-27-voo1", "2025-01-28"])
    ap.add_argument("--val", nargs="*", default=["2025-01-27-voo2"])
    ap.add_argument("--teste", nargs="*", default=[])
    ap.add_argument("--escala", type=float, default=0.5,
                    help="fator de redução da foto antes de recortar (1 = original)")
    ap.add_argument("--recorte", type=int, default=256)
    ap.add_argument("--sobreposicao", type=int, default=12)
    ap.add_argument("--a-cada", type=int, default=1,
                    help="usa um disparo a cada N (reduz a redundância entre fotos seguidas)")
    args = ap.parse_args(argv)

    require(args.origem, "a pasta das fotos do Cerrado",
            "Confira --origem (no LISA: " + ORIGEM + ").")
    splits = {"train": args.treino, "val": args.val, "test": args.teste}
    summary = build_dataset(args.origem, args.saida, splits, args.escala, args.recorte,
                            args.sobreposicao, args.a_cada)
    for split, s in summary["divisoes"].items():
        print(f"{split}: {s['recortes']} recortes de {', '.join(s['voos']) or '—'}")
    print(f"Gravado em {args.saida}/ (manifesto.json com os detalhes)")
    return summary


if __name__ == "__main__":
    main()
