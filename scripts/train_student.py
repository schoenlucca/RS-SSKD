"""Treino completo do aluno, etapa 1 (item 5).

Uso:
    python scripts/train_student.py --nome lambda1
    python scripts/train_student.py --nome lambda0   --peso-consistencia 0
    python scripts/train_student.py --nome lambda100 --peso-consistencia 100
    python scripts/train_student.py --nome lambda1 --continuar   # retoma

O que acontece em cada época:
    1. as 476 imagens de treino são embaralhadas e lidas em lotes, cada uma
       numa orientação sorteada (aumento de dados D4, parte 2c);
    2. em cada lote, um passo de treino com a perda da etapa 1 (parte 4c):
           L = L_MaskRCNN(copas do TEBAKA) + lambda * consistência RS-SSKD;
    3. no fim da época, a média de cada perda vai para o histórico, e o
       modelo é salvo (ultimo.pt), para poder retomar se o treino parar.

A taxa de aprendizado sobe nos primeiros passos (aquecimento) e depois desce
em forma de cosseno até o fim do treino (rsskd/train.py, lr_schedule).

Saídas em runs/<nome>/ (pasta ignorada pelo git):
    config.json     todos os parâmetros do treino
    historico.csv   uma linha por época, com a média de cada perda
    ultimo.pt       pesos do modelo e estado do otimizador (~500 MB)

A avaliação na validação, o melhor modelo e a parada antecipada vêm na
parte 5b.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.augment import RandomDihedral, make_loader  # noqa: E402
from rsskd.data import DEFAULT_DIR  # noqa: E402
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.detector import build_student  # noqa: E402
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.train import lr_schedule, make_optimizer, pick_device, train_step  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True, help="nome do treino (pasta em runs/)")
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    ap.add_argument("--copas", default="data/pseudo/agml")
    ap.add_argument("--entrada", default="configs/agml_entrada.json")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--aquecimento", type=int, default=200,
                    help="passos de aquecimento da taxa de aprendizado")
    ap.add_argument("--consistencia", default="artigo",
                    help="artigo (padrão), sorteio, nenhuma ou lista (ex.: flip,rot90)")
    ap.add_argument("--peso-consistencia", type=float, default=1.0,
                    help="lambda da consistência; 0 desliga")
    ap.add_argument("--sem-aumento", action="store_true",
                    help="desliga as rotações/espelhamentos sorteados")
    ap.add_argument("--sem-pretreino", action="store_true")
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", default="runs")
    ap.add_argument("--continuar", action="store_true",
                    help="retoma do ultimo.pt, se existir")
    ap.add_argument("--max-passos-por-epoca", type=int, default=None,
                    help="limita os passos por época (para testes rápidos)")
    return ap.parse_args(argv)


def set_seed(seed: int):
    """Fixa as sementes para o treino poder ser repetido."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.saida) / args.nome
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.semente)
    device = pick_device(args.dispositivo)
    consistency = (None if args.consistencia == "nenhuma" or args.peso_consistencia == 0
                   else args.consistencia)

    # Dados: treino com as copas do TEBAKA como alvo e aumento de dados.
    spec = InputSpec.load(args.entrada)
    train_ds = CrownDataset("train", args.copas, spec, dataset_dir=args.pasta,
                            transform=None if args.sem_aumento else RandomDihedral())
    loader = make_loader(train_ds, batch_size=args.lote, shuffle=True)
    steps_per_epoch = len(loader)
    if args.max_passos_por_epoca:
        steps_per_epoch = min(steps_per_epoch, args.max_passos_por_epoca)
    total_steps = steps_per_epoch * args.epocas
    image_size = int(train_ds[0][0].shape[-1])

    model = build_student(spec.in_channels, not args.sem_pretreino,
                          image_size=image_size).to(device)
    opt = make_optimizer(model, lr=args.lr)

    # Retomar um treino interrompido.
    start_epoch, step = 0, 0
    ckpt_path = run_dir / "ultimo.pt"
    if args.continuar and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["modelo"])
        opt.load_state_dict(ckpt["otimizador"])
        start_epoch, step = ckpt["epoca"] + 1, ckpt["passo"]
        torch.set_rng_state(ckpt["rng"])
        print(f"Retomando de {ckpt_path}, época {start_epoch}")
    elif (run_dir / "historico.csv").exists():
        (run_dir / "historico.csv").unlink()   # treino novo: histórico novo

    config = vars(args) | {"dispositivo_usado": str(device),
                           "imagens_treino": len(train_ds),
                           "passos_por_epoca": steps_per_epoch,
                           "passos_total": total_steps}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"Treino '{args.nome}': {len(train_ds)} imagens, {args.epocas} épocas de "
          f"{steps_per_epoch} passos, dispositivo {device}, consistência "
          f"{consistency or 'desligada'} (lambda {args.peso_consistencia})")

    for epoch in range(start_epoch, args.epocas):
        start, sums, n, lr = time.time(), {}, 0, args.lr
        for images, targets in loader:
            if n >= steps_per_epoch:
                break
            lr = lr_schedule(opt, step, total_steps, args.aquecimento, args.lr)
            losses = train_step(model, opt, images, targets, device,
                                consistency=consistency,
                                consistency_weight=args.peso_consistencia)
            for k, v in losses.items():
                if isinstance(v, float):
                    sums[k] = sums.get(k, 0.0) + v
            sums["pulados"] = sums.get("pulados", 0) + int(losses["pulado"])
            n, step = n + 1, step + 1
            if n % 50 == 0:
                print(f"  época {epoch} passo {n}/{steps_per_epoch}  "
                      f"perda {losses['total']:.3f}  lr {lr:.5f}", flush=True)

        # Médias da época no histórico.
        row = {"epoca": epoch, "lr_final": lr, "tempo_s": round(time.time() - start)}
        row |= {k: (v if k == "pulados" else v / n) for k, v in sums.items()}
        hist = run_dir / "historico.csv"
        new = not hist.exists()
        with open(hist, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        cons = (f"  consistência {row['loss_consistencia']:.4f}"
                if "loss_consistencia" in row else "")
        print(f"Época {epoch}: perda {row['total']:.3f}  máscara {row['loss_mask']:.3f}"
              f"{cons}  ({row['tempo_s']} s)", flush=True)

        torch.save({"modelo": model.state_dict(), "otimizador": opt.state_dict(),
                    "epoca": epoch, "passo": step, "rng": torch.get_rng_state(),
                    "config": config}, ckpt_path)

    print(f"Fim. Histórico e modelo em {run_dir}/")
    return run_dir


if __name__ == "__main__":
    main()
