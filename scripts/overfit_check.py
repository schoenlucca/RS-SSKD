"""Checagem de sobreajuste: o aluno consegue imitar o professor em poucas imagens?

Uso:
    python scripts/overfit_check.py
    python scripts/overfit_check.py --n-imagens 4 --passos 300
    python scripts/overfit_check.py --consistencia nenhuma      # só o professor
    python scripts/overfit_check.py --consistencia sorteio --peso-consistencia 10

Por padrão, treina com a perda da etapa 1 completa: copas do TEBAKA mais a
consistência do RS-SSKD (as 3 transformações do artigo, lambda = 1).

**A ideia.** Antes de gastar horas de GPU num treino completo, treinamos o
aluno só em 4 imagens, sem aumento de dados, por algumas centenas de passos.
Com tão poucas imagens, um modelo que funciona tem de conseguir "decorar"
as copas do professor: a perda cai perto de zero e, nessas mesmas imagens,
as copas previstas coincidem com as do professor (F1 alto contra o
professor). Se isso não acontecer, há um erro no caminho (dados, alvos,
modelo ou otimizador) e um treino longo não resolveria.

Isso **não** mede se o aluno é bom: medido nas imagens que decorou, ele
parece ótimo. A comparação com a referência aparece só como curiosidade; a
avaliação de verdade é na validação, no item 5.

Saídas em saidas/sobreajuste/:
    perdas.png    curva das perdas a cada passo
    copas.png     imagem, copas do professor e copas do aluno
    resumo.json   parâmetros e métricas

Tempo: na CPU, cerca de 2 a 5 s por passo (lote de 2). 200 passos levam de
7 a 15 min; numa GPU, menos de 1 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.augment import make_loader  # noqa: E402
from rsskd.data import DEFAULT_DIR  # noqa: E402
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.detector import build_student  # noqa: E402
from rsskd.evaluation import CrownMetrics, format_summary, predictions_to_instances  # noqa: E402
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.train import make_optimizer, pick_device, train_step, warmup_lr  # noqa: E402
from show_batch import rgb_from_input  # noqa: E402


def colorize(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    colors = np.random.default_rng(seed).random((labels.max() + 1, 3))
    colors[0] = 0
    return colors[labels]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    ap.add_argument("--copas", default="data/pseudo/agml")
    ap.add_argument("--entrada", default="configs/agml_entrada.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-imagens", type=int, default=4)
    ap.add_argument("--passos", type=int, default=200)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--sem-pretreino", action="store_true",
                    help="não usa os pesos do ImageNet (mais difícil de convergir)")
    ap.add_argument("--dispositivo", default="auto", help="auto, cpu ou cuda")
    ap.add_argument("--consistencia", default="artigo",
                    help="artigo (padrão), sorteio, nenhuma ou lista (ex.: flip,rot90)")
    ap.add_argument("--peso-consistencia", type=float, default=1.0,
                    help="lambda da consistência do RS-SSKD (1 no artigo)")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--saida", default="saidas/sobreajuste")
    args = ap.parse_args(argv)

    torch.manual_seed(args.semente)
    device = pick_device(args.dispositivo)
    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Poucas imagens, sem aumento de dados (para o aluno poder decorá-las).
    spec = InputSpec.load(args.entrada)
    ds = CrownDataset(args.split, args.copas, spec, dataset_dir=args.pasta)
    rng = np.random.default_rng(args.semente)
    idx = sorted(rng.choice(len(ds), size=min(args.n_imagens, len(ds)),
                            replace=False).tolist())
    subset = torch.utils.data.Subset(ds, idx)
    image_size = int(ds[idx[0]][0].shape[-1])

    # 2. Aluno e otimizador.
    model = build_student(spec.in_channels, not args.sem_pretreino,
                          image_size=image_size).to(device)
    opt = make_optimizer(model, lr=args.lr)
    warmup = min(50, args.passos // 4)
    consistency = None if args.consistencia == "nenhuma" else args.consistencia
    print(f"Imagens {idx} ({args.split}), {args.passos} passos, lote {args.lote}, "
          f"dispositivo {device}, pré-treino {'não' if args.sem_pretreino else 'sim'}, "
          f"consistência {args.consistencia} (lambda {args.peso_consistencia})")

    # 3. Treino: percorre as poucas imagens repetidamente.
    history, step, start = [], 0, time.time()
    while step < args.passos:
        for images, targets in make_loader(subset, batch_size=args.lote, shuffle=True):
            if step >= args.passos:
                break
            lr = warmup_lr(opt, step, warmup, args.lr)
            losses = train_step(model, opt, images, targets, device,
                                consistency=consistency,
                                consistency_weight=args.peso_consistencia)
            losses.update(passo=step, lr=lr)
            history.append(losses)
            if step % 10 == 0 or step == args.passos - 1:
                cons = (f"  consistência {losses['loss_consistencia']:.4f}"
                        if "loss_consistencia" in losses else "")
                print(f"passo {step:4d}  perda {losses['total']:.3f}  "
                      f"(máscara {losses['loss_mask']:.3f}){cons}  lr {lr:.4f}  "
                      f"{time.time() - start:.0f} s", flush=True)
            step += 1

    # 4. Avaliação nas mesmas imagens: contra o professor (deve ser alta) e
    #    contra a referência (só como curiosidade).
    model.eval()
    vs_teacher, vs_ref = CrownMetrics(), CrownMetrics()
    rows = []
    with torch.no_grad():
        for i in idx:
            image, _ = ds[i]
            (out,) = model([image.to(device)])
            pred = predictions_to_instances(out)
            teacher = ds.store.load(args.split, i)
            ref = ds.reference(i)
            vs_teacher.add(pred, teacher, index=i)
            vs_ref.add(pred, ref, index=i)
            rows.append((image, teacher, pred, ref))
    s_t, s_r = vs_teacher.summary(), vs_ref.summary()
    print(f"\nAluno x professor (nas imagens treinadas): {format_summary(s_t)}")
    print(f"Aluno x referência (só curiosidade):        {format_summary(s_r)}")

    # 5. Figuras e resumo.
    fig, ax = plt.subplots(figsize=(8, 4))
    passos = [h["passo"] for h in history]
    for key, label in [("total", "total"), ("loss_mask", "máscaras"),
                       ("loss_classifier", "classificação"), ("loss_box_reg", "caixas"),
                       ("loss_objectness", "RPN"), ("loss_consistencia", "consistência")]:
        if key in history[0]:
            ax.plot(passos, [h[key] for h in history], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("passo")
    ax.set_ylabel("perda")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "perdas.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    for r, (i, (image, teacher, pred, ref)) in enumerate(zip(idx, rows)):
        ax[r, 0].imshow(rgb_from_input(image, spec))
        ax[r, 0].set_title(f"imagem {i}", fontsize=9)
        ax[r, 1].imshow(colorize(teacher))
        ax[r, 1].set_title(f"professor: {teacher.max()} copas", fontsize=9)
        ax[r, 2].imshow(colorize(pred))
        ax[r, 2].set_title(f"aluno: {pred.max()} copas", fontsize=9)
        ax[r, 3].imshow(colorize(ref))
        ax[r, 3].set_title(f"referência: {ref.max()} copas", fontsize=9)
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "copas.png", dpi=100)
    plt.close(fig)

    summary = {"imagens": idx, "passos": args.passos, "lote": args.lote,
               "lr": args.lr, "pretreino": not args.sem_pretreino,
               "dispositivo": str(device), "tempo_s": round(time.time() - start),
               "consistencia": args.consistencia,
               "peso_consistencia": args.peso_consistencia,
               "consistencia_inicial": history[0].get("loss_consistencia"),
               "consistencia_final": history[-1].get("loss_consistencia"),
               "perda_inicial": history[0]["total"], "perda_final": history[-1]["total"],
               "passos_pulados": sum(h["pulado"] for h in history),
               "aluno_x_professor": s_t, "aluno_x_referencia": s_r}
    (out_dir / "resumo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Figuras e resumo em {out_dir}/")
    return summary


if __name__ == "__main__":
    main()
