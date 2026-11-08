"""Treino completo do aluno: etapa 1 (item 5) e etapa 2 (parte 6d).

Uso, etapa 1:
    python scripts/train_student.py --nome lambda1
    python scripts/train_student.py --nome lambda0   --peso-consistencia 0
    python scripts/train_student.py --nome lambda100 --peso-consistencia 100
    python scripts/train_student.py --nome lambda1 --continuar   # retoma

Uso, etapa 2 (com um aluno da etapa 1 como professor):
    python scripts/train_student.py --nome etapa2 --professor runs/lambda100/melhor.pt
    python scripts/train_student.py --nome etapa2_regra --professor runs/lambda100/melhor.pt \
        --mu-max 0.3
    python scripts/train_student.py --nome so_regra --mu-max 0.3     # regra sem destilação

Na etapa 2, a perda do passo é (rsskd/train.py, Stage2):
    (1 - mu) * classificação + demais perdas do Mask R-CNN    (TEBAKA)
    + mu * regra 1 de Harmon        (se --mu-max > 0)
    + beta * auto-destilação        (se --professor; beta = --peso-kd)
    + lambda * consistência         (como na etapa 1)
O aluno novo começa dos pesos do ImageNet (born-again); --iniciar-do-professor
começa dos pesos do professor. A área de referência da regra vem das caixas
das copas do TEBAKA no treino, calculada no início: a mediana (padrão), a
média ou um percentil (--estat-area p90), que deixa a regra pesar só sobre as
copas grandes demais.

O que acontece em cada época:
    1. as 476 imagens de treino são embaralhadas e lidas em lotes, cada uma
       numa orientação sorteada (aumento de dados D4, parte 2c);
    2. em cada lote, um passo de treino com a perda da etapa 1 (parte 4c):
           L = L_MaskRCNN(copas do TEBAKA) + lambda * consistência RS-SSKD;
    3. no fim da época, o aluno é avaliado nas 59 imagens de validação
       (parte 5b), contra a referência e contra as copas do professor;
    4. a média de cada perda e as métricas vão para o histórico, e o modelo
       é salvo (ultimo.pt), para poder retomar se o treino parar;
    5. se for o melhor até agora pelo critério escolhido, é salvo também em
       melhor.pt; se passar `--paciencia` épocas sem melhorar, o treino para
       (parada antecipada).

**Critério para escolher o melhor modelo (`--criterio`).**
    professor  (padrão) F1 do aluno contra as copas do TEBAKA na validação.
               Não usa nenhum rótulo humano, coerente com a proposta do
               projeto e aplicável às imagens do drone, que não terão
               referência.
    referencia F1 contra as máscaras de referência da validação. Escolhe
               melhor para o AgML, mas usa rótulos humanos para decidir,
               o que precisa ser declarado no relatório.
    nenhum     sem escolha nem parada: o modelo final é o da última época.
As métricas contra a referência são sempre registradas, qualquer que seja o
critério; o conjunto de teste nunca é usado aqui.

A taxa de aprendizado sobe nos primeiros passos (aquecimento) e depois desce
em forma de cosseno até o fim do treino (rsskd/train.py, lr_schedule).

Saídas em runs/<nome>/ (pasta ignorada pelo git):
    config.json     todos os parâmetros do treino
    historico.csv   uma linha por época: média de cada perda e métricas na
                    validação (val_* contra a referência, valprof_* contra o
                    professor) e se foi o melhor até então
    ultimo.pt       pesos do modelo e estado do otimizador (~500 MB)
    melhor.pt       pesos do melhor modelo pelo critério (~250 MB)
    melhor_referencia.pt
                    pesos da época de maior F1 contra a referência, qualquer
                    que seja o critério. SÓ PARA DIAGNÓSTICO: escolher um
                    modelo por ele usa rótulos humanos; se for usado no
                    relatório, isso tem de ser declarado (~250 MB)
    epocas/epoca_NN.pt
                    com --salvar-epocas: os pesos de cada época pedida
                    (~250 MB cada), para escolher depois, por exemplo, o aluno
                    de uma época inicial como professor da etapa 2

**Por que guardar épocas.** Nos treinos da etapa 1, o F1 contra a
referência sobe nas primeiras épocas e depois cai, enquanto o F1 contra o
professor só sobe: o aluno aprende primeiro o que é uma copa e depois passa a
decorar os erros do TEBAKA (o fenômeno de "early learning" com rótulos
ruidosos). O aluno de uma época inicial pode ser um professor melhor para a
etapa 2 do que o da última.
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
from rsskd.evaluation import CrownMetrics, evaluate_student, format_summary  # noqa: E402
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.distill import freeze  # noqa: E402
from rsskd.rules import check_area_stat, reference_area  # noqa: E402
from rsskd.train import (  # noqa: E402
    Stage2, load_checkpoint, lr_schedule, make_optimizer, mu_schedule, pick_device,
    train_step,
)


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
    ap.add_argument("--criterio", default="professor",
                    choices=["professor", "referencia", "nenhum"],
                    help="como escolher o melhor modelo na validação")
    ap.add_argument("--salvar-epocas", type=int, nargs="*", default=None,
                    help="guarda os pesos destas épocas em epocas/ (sem números: todas)")
    ap.add_argument("--paciencia", type=int, default=6,
                    help="épocas sem melhorar antes de parar (0 desliga)")
    # Etapa 2
    ap.add_argument("--professor", default=None,
                    help="melhor.pt (ou ultimo.pt) da etapa 1: liga a auto-destilação")
    ap.add_argument("--peso-kd", type=float, default=1.0, help="beta da auto-destilação")
    ap.add_argument("--temperatura", type=float, default=4.0)
    ap.add_argument("--peso-kd-mascara", type=float, default=1.0,
                    help="peso da KL das máscaras dentro da auto-destilação")
    ap.add_argument("--iniciar-do-professor", action="store_true",
                    help="aluno começa com os pesos do professor (padrão: ImageNet)")
    ap.add_argument("--mu-max", type=float, default=0.0,
                    help="mu final da regra 1 de Harmon; 0 desliga a regra")
    ap.add_argument("--mu-inicio", type=float, default=0.0,
                    help="fração do treino antes de mu começar a subir")
    ap.add_argument("--area-regra", type=float, default=None,
                    help="área de referência (px); padrão: calculada das copas do TEBAKA")
    ap.add_argument("--estat-area", default="mediana", type=_area_stat,
                    help="mediana, media ou p<N>, um percentil das áreas (ex.: p90)")
    ap.add_argument("--regra-k", type=float, default=5.0)
    ap.add_argument("--regra-C", type=float, default=1.0)
    ap.add_argument("--regra-lambda", type=float, default=1.0)
    ap.add_argument("--max-passos-por-epoca", type=int, default=None,
                    help="limita os passos por época (para testes rápidos)")
    return ap.parse_args(argv)


def _area_stat(value: str) -> str:
    try:
        return check_area_stat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def append_history(path: Path, row: dict) -> None:
    """Acrescenta uma linha ao histórico. Se a linha trouxer uma coluna nova
    (por exemplo, a regra, que só entra quando mu > 0), o arquivo é reescrito
    com todas as colunas, para nenhum valor cair na coluna errada."""
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


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
    # Validação: sem aumento de dados, para coincidir com a referência.
    val_ds = CrownDataset("val", args.copas, spec, dataset_dir=args.pasta)
    steps_per_epoch = len(loader)
    if args.max_passos_por_epoca:
        steps_per_epoch = min(steps_per_epoch, args.max_passos_por_epoca)
    total_steps = steps_per_epoch * args.epocas
    image_size = int(train_ds[0][0].shape[-1])

    model = build_student(spec.in_channels, not args.sem_pretreino,
                          image_size=image_size).to(device)

    # Etapa 2: professor congelado e área de referência da regra.
    stage2, extra = None, {}
    if args.professor or args.mu_max > 0:
        teacher = None
        if args.professor:
            loaded, t_cfg, t_ckpt = load_checkpoint(args.professor, device)
            if InputSpec.load(t_cfg["entrada"]).in_channels != spec.in_channels:
                raise ValueError("O professor foi treinado com outra entrada.")
            teacher = freeze(loaded)
            if args.iniciar_do_professor:
                model.load_state_dict(t_ckpt["modelo"])
            extra["professor_epoca"] = t_ckpt["epoca"]
            print(f"Professor: {args.professor} (época {t_ckpt['epoca']}), congelado")
        area = args.area_regra
        if args.mu_max > 0 and area is None:
            store = train_ds.store
            area = reference_area((store.load("train", i) for i in store.indices("train")),
                                  args.estat_area)
            print(f"Área de referência da regra 1 ({args.estat_area} das caixas do "
                  f"TEBAKA no treino): {area:.0f} px")
        extra["area_regra_usada"] = area
        stage2 = Stage2(teacher=teacher, kd_weight=args.peso_kd,
                        temperature=args.temperatura, mask_weight=args.peso_kd_mascara,
                        rule_area=area if args.mu_max > 0 else None, rule_k=args.regra_k,
                        rule_C=args.regra_C, rule_lam=args.regra_lambda)
    opt = make_optimizer(model, lr=args.lr)

    # Retomar um treino interrompido.
    start_epoch, step = 0, 0
    best_score, best_epoch = -1.0, -1
    best_ref = -1.0
    ckpt_path = run_dir / "ultimo.pt"
    if args.continuar and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["modelo"])
        opt.load_state_dict(ckpt["otimizador"])
        start_epoch, step = ckpt["epoca"] + 1, ckpt["passo"]
        torch.set_rng_state(ckpt["rng"])
        best_score, best_epoch = ckpt.get("melhor_score", -1.0), ckpt.get("melhor_epoca", -1)
        best_ref = ckpt.get("melhor_ref_score", -1.0)
        print(f"Retomando de {ckpt_path}, época {start_epoch}")
    elif (run_dir / "historico.csv").exists():
        (run_dir / "historico.csv").unlink()   # treino novo: histórico novo

    config = vars(args) | extra | {"etapa": 2 if stage2 else 1,
                                   "dispositivo_usado": str(device),
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
            if stage2 is not None:
                stage2.mu = mu_schedule(step, total_steps, args.mu_max, args.mu_inicio)
            losses = train_step(model, opt, images, targets, device,
                                consistency=consistency,
                                consistency_weight=args.peso_consistencia,
                                stage2=stage2)
            for k, v in losses.items():
                if isinstance(v, float):
                    sums[k] = sums.get(k, 0.0) + v
            sums["pulados"] = sums.get("pulados", 0) + int(losses["pulado"])
            n, step = n + 1, step + 1
            if n % 50 == 0:
                print(f"  época {epoch} passo {n}/{steps_per_epoch}  "
                      f"perda {losses['total']:.3f}  lr {lr:.5f}", flush=True)

        # Médias da época.
        row = {"epoca": epoch, "lr_final": lr, "tempo_s": round(time.time() - start)}
        row |= {k: (v if k == "pulados" else v / n) for k, v in sums.items()}

        # Avaliação na validação: contra a referência e contra o professor,
        # numa passada só.
        vs_teacher = CrownMetrics()
        s_ref, _ = evaluate_student(model, val_ds, device, teacher_metrics=vs_teacher)
        s_prof = vs_teacher.summary()
        row |= {f"val_{k}": v for k, v in s_ref.items()}
        row |= {f"valprof_{k}": v for k, v in s_prof.items()}

        # Melhor modelo e parada antecipada.
        score = {"professor": s_prof["f1"], "referencia": s_ref["f1"],
                 "nenhum": float(epoch)}[args.criterio]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save({"modelo": model.state_dict(), "epoca": epoch,
                        "criterio": args.criterio, "score": score,
                        "val_referencia": s_ref, "val_professor": s_prof,
                        "config": config}, run_dir / "melhor.pt")
        row["melhor"] = int(improved)

        # Diagnóstico: a época de maior F1 contra a referência.
        if s_ref["f1"] > best_ref:
            best_ref = s_ref["f1"]
            torch.save({"modelo": model.state_dict(), "epoca": epoch,
                        "criterio": "referencia (diagnóstico)", "score": best_ref,
                        "val_referencia": s_ref, "val_professor": s_prof,
                        "config": config}, run_dir / "melhor_referencia.pt")
        # Épocas pedidas em --salvar-epocas (lista vazia = todas).
        if args.salvar_epocas is not None and (not args.salvar_epocas
                                               or epoch in args.salvar_epocas):
            (run_dir / "epocas").mkdir(exist_ok=True)
            torch.save({"modelo": model.state_dict(), "epoca": epoch,
                        "val_referencia": s_ref, "val_professor": s_prof,
                        "config": config}, run_dir / "epocas" / f"epoca_{epoch:02d}.pt")
        append_history(run_dir / "historico.csv", row)
        cons = (f"  consistência {row['loss_consistencia']:.4f}"
                if "loss_consistencia" in row else "")
        if "loss_kd" in row:
            cons += f"  destilação {row['loss_kd']:.4f}"
        if "loss_regra1" in row:
            cons += (f"  regra1 {row['loss_regra1']:.4f} (satisfeita "
                     f"{row['regra1_satisfeita']:.2f}, mu {row['mu']:.2f})")
        print(f"Época {epoch}: perda {row['total']:.3f}  máscara {row['loss_mask']:.3f}"
              f"{cons}  ({row['tempo_s']} s)\n"
              f"  validação x referência: {format_summary(s_ref)}\n"
              f"  validação x professor:  {format_summary(s_prof)}"
              f"{'   <- melhor' if improved else ''}", flush=True)

        torch.save({"modelo": model.state_dict(), "otimizador": opt.state_dict(),
                    "epoca": epoch, "passo": step, "rng": torch.get_rng_state(),
                    "melhor_score": best_score, "melhor_epoca": best_epoch,
                    "melhor_ref_score": best_ref, "config": config}, ckpt_path)

        if (args.criterio != "nenhum" and args.paciencia > 0
                and epoch - best_epoch >= args.paciencia):
            print(f"Parada antecipada: {args.paciencia} épocas sem melhorar "
                  f"(melhor: época {best_epoch}).")
            break

    print(f"Fim. Melhor modelo: época {best_epoch} ({args.criterio}). "
          f"Histórico e modelos em {run_dir}/")
    return run_dir


if __name__ == "__main__":
    main()
