"""Peças do treino do aluno, usadas pela checagem de sobreajuste (parte 3c)
e, depois, pelo laço de treino completo (item 5).

Um passo de treino, no PyTorch:

    1. o modelo recebe as imagens e os alvos (copas do professor) e devolve
       as 5 perdas do Mask R-CNN;
    2. somamos as perdas e calculamos os gradientes (backward): para cada
       peso, em que direção ele deve mudar para a perda diminuir;
    3. o otimizador (SGD com momento) dá um passo nessa direção, do tamanho
       da taxa de aprendizado.

Dois cuidados, motivados pelo que vimos na parte 3a (a perda das máscaras
disparou com taxa 0,005 numa rede sem pré-treino):

  - **aquecimento (warm-up):** a taxa de aprendizado começa pequena e sobe
    linearmente até o valor escolhido nos primeiros passos. No começo as
    cabeças do Mask R-CNN são aleatórias e os gradientes, grandes e
    desencontrados; passos grandes nessa hora desestabilizam a rede.
  - **limite do gradiente (clipping):** se a norma do gradiente passar de
    um limite, ele é reduzido proporcionalmente. A direção do passo é a
    mesma; só o tamanho fica limitado.

Com a consistência do RS-SSKD ligada (parte 4c), a perda do passo é a
perda da etapa 1 do aluno:

    L = L_MaskRCNN(imagem original, copas do TEBAKA)
        + lambda * soma_T L_ss(T)            (rsskd/consistency.py)

Como no artigo, a perda supervisionada (aqui, a do professor TEBAKA) usa só
a imagem original; as imagens transformadas entram só na consistência.

Na etapa 2 (parte 6c), o passo recebe também um `Stage2`, e a perda segue a
equação 16 de Harmon et al. (2022), somada à auto-destilação do RS-SSKD:

    L = (1 - mu) * L_classificação        (copas do TEBAKA)
        + demais perdas do Mask R-CNN      (caixas, máscaras, RPN; TEBAKA)
        + mu * L_regra1                    (professor por regras, Harmon)
        + beta * L_auto-destilação         (aluno da etapa 1 congelado)
        + lambda * consistência            (como na etapa 1)

Em Harmon, (1 - mu) multiplica só a classificação: é a cabeça em que a
regra age; as outras perdas continuam inteiras. mu cresce com o passo
(mu_schedule), para a regra só pesar depois que o aluno já aprendeu o
básico: o professor das regras é o próprio aluno corrigido, então ele só é
bom quando o aluno já é razoável.
"""

from __future__ import annotations

from pathlib import Path

import math
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from .consistency import capture_objectness, choose_transforms, consistency_loss
from .distill import capture_roi, distillation_loss
from .rules import rules_loss_from_capture


@dataclass
class Stage2:
    """Configuração da etapa 2 para um passo de treino.

    teacher:       aluno da etapa 1 congelado (distill.freeze); None = sem
                   auto-destilação.
    kd_weight:     beta, peso da auto-destilação.
    temperature:   T da KL (4, como no RS-SSKD).
    mask_weight:   peso da KL das máscaras dentro da auto-destilação.
    mu:            peso da regra neste passo (vem de mu_schedule).
    rule_area:     área de referência da regra 1 (rules.reference_area);
                   None = sem regra.
    rule_k, rule_C, rule_lam: parâmetros da regra (k, C e lambda de Harmon).
    """

    teacher: torch.nn.Module | None = None
    kd_weight: float = 1.0
    temperature: float = 4.0
    mask_weight: float = 1.0
    mu: float = 0.0
    rule_area: float | None = None
    rule_k: float = 5.0
    rule_C: float = 1.0
    rule_lam: float = 1.0


def mu_schedule(step: int, total_steps: int, mu_max: float,
                start_frac: float = 0.0) -> float:
    """mu da equação 16 de Harmon: "uma função do número do passo em [0, 1]".

    Fica em 0 até start_frac do treino e sobe em linha reta até mu_max no
    último passo. Com start_frac = 0, a subida começa no primeiro passo.
    """
    start = start_frac * total_steps
    if step <= start or total_steps <= start:
        return 0.0
    return min(mu_max, mu_max * (step - start) / max(1.0, total_steps - start))


def load_checkpoint(path, device: "str | torch.device" = "cpu"):
    """Recria o aluno salvo por scripts/train_student.py e carrega os pesos.

    Devolve (modelo em modo de avaliação, config do treino, checkpoint).
    Funciona com melhor.pt e com ultimo.pt.
    """
    from .detector import build_student
    from .inputs import InputSpec
    from .paths import checkpoints_near, require

    path = Path(path)
    if not path.exists():
        runs = path.parent.parent
        others = checkpoints_near(path) or (
            sorted(d.name for d in runs.iterdir() if d.is_dir()) if runs.exists() else [])
        require(path, "o modelo salvo",
                "Confira o --run e o --modelo (melhor, ultimo, melhor_referencia ou "
                "epocas/epoca_NN). Um treino novo grava melhor.pt ao fim da 1ª época.",
                others)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    spec = InputSpec.load(cfg["entrada"])
    model = build_student(spec.in_channels, not cfg["sem_pretreino"])
    model.load_state_dict(ckpt["modelo"])
    return model.to(device).eval(), cfg, ckpt


def pick_device(name: str = "auto") -> torch.device:
    """'auto' usa a GPU (cuda) se houver; senão, a CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def make_optimizer(model: torch.nn.Module, lr: float = 0.005,
                   momentum: float = 0.9, weight_decay: float = 1e-4):
    """SGD só sobre os parâmetros que treinam (os congelados ficam de fora).

    São os valores usuais de treino do Mask R-CNN no torchvision. O
    weight_decay puxa os pesos levemente para zero, contra o sobreajuste.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=momentum,
                           weight_decay=weight_decay)


def warmup_lr(optimizer, step: int, warmup_steps: int, base_lr: float) -> float:
    """Ajusta a taxa de aprendizado do passo `step` (contando de 0).

    Nos primeiros `warmup_steps` passos ela sobe de base_lr/10 até base_lr;
    depois, fica em base_lr. Devolve a taxa usada.
    """
    if warmup_steps > 0 and step < warmup_steps:
        frac = step / warmup_steps
        lr = base_lr * (0.1 + 0.9 * frac)
    else:
        lr = base_lr
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def lr_schedule(optimizer, step: int, total_steps: int, warmup_steps: int,
                base_lr: float, min_ratio: float = 0.01) -> float:
    """Taxa de aprendizado do treino completo: aquecimento + decaimento cosseno.

        aquecimento (passos 0 .. warmup):   sobe de base_lr/10 até base_lr;
        depois, até o último passo:         desce suavemente, em forma de
                                            meia onda de cosseno, até
                                            base_lr * min_ratio.

    Passos grandes no meio do treino exploram; passos pequenos no fim
    "assentam" os pesos num bom ponto, sem ficar pulando em volta dele.
    Devolve a taxa usada.
    """
    if step < warmup_steps:
        return warmup_lr(optimizer, step, warmup_steps, base_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    lr = base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def to_device(images, targets, device):
    """Leva o lote para a GPU (ou CPU)."""
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_step(model, optimizer, images, targets, device,
               max_grad_norm: float = 10.0, consistency: str | None = None,
               consistency_weight: float = 1.0, stage2: Stage2 | None = None) -> dict:
    """Um passo de treino. Devolve as perdas (números) e a perda total.

    consistency:        None (desligada), "artigo" (as 3 transformações do
                        RS-SSKD em todo passo), "sorteio" (uma do D4) ou uma
                        lista de nomes; ver choose_transforms.
    consistency_weight: o lambda que multiplica a consistência (1 no artigo).

    stage2:             None na etapa 1; um Stage2 na etapa 2.

    Além das 5 perdas do Mask R-CNN, devolve "loss_consistencia" (sem o
    lambda) e "cons_<transformação>" para cada transformação usada; na
    etapa 2, também "loss_kd" (sem o beta), "kd_cls", "kd_mask", "mu",
    "loss_regra1" (sem o mu) e "regra1_satisfeita".

    Se alguma perda virar infinita ou NaN (sinal de treino instável), o passo
    é pulado em vez de estragar os pesos, e "pulado" vem True.
    """
    model.train()
    images, targets = to_device(images, targets, device)
    use_roi = stage2 is not None and (stage2.teacher is not None
                                      or stage2.rule_area is not None)
    # Os ganchos só escutam o passo normal; a consistência roda depois,
    # fora deles, para não sobrescrever o que foi guardado.
    with ExitStack() as stack:
        cap = stack.enter_context(capture_objectness(model)) if consistency else {}
        roi = stack.enter_context(capture_roi(model)) if use_roi else {}
        losses = model(images, targets)            # perda do professor TEBAKA
    out = {k: float(v.detach()) for k, v in losses.items()}
    if stage2 is not None:
        # Equação 16 de Harmon: (1 - mu) só na classificação.
        losses = dict(losses)
        losses["loss_classifier"] = (1 - stage2.mu) * losses["loss_classifier"]
    total = torch.stack(list(losses.values())).sum()

    if stage2 is not None and stage2.teacher is not None:
        kd, parts = distillation_loss(roi, stage2.teacher, stage2.temperature,
                                      stage2.mask_weight)
        total = total + stage2.kd_weight * kd
        out["loss_kd"] = float(kd.detach())
        out.update(parts)
    if stage2 is not None and stage2.rule_area is not None:
        # Calculada mesmo com mu = 0 (no começo do treino), para o histórico
        # mostrar a regra desde o início; com mu = 0 ela não pesa.
        rule, parts = rules_loss_from_capture(roi, stage2.rule_area, stage2.rule_k,
                                              stage2.rule_C, stage2.rule_lam)
        total = total + stage2.mu * rule
        out["loss_regra1"] = parts["regra1"]
        out["regra1_satisfeita"] = parts["regra1_satisfeita"]
    if stage2 is not None:
        out["mu"] = stage2.mu

    if consistency:
        # As imagens do lote têm o mesmo tamanho: empilhamos num tensor só.
        cons, parts = consistency_loss(model, torch.stack(images),
                                       cap["objectness"], choose_transforms(consistency))
        total = total + consistency_weight * cons
        out["loss_consistencia"] = float(cons.detach())
        out.update({f"cons_{k}": v for k, v in parts.items()})

    out["total"] = float(total.detach())
    if not torch.isfinite(total):
        optimizer.zero_grad(set_to_none=True)
        out["pulado"] = True
        return out
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm)
    optimizer.step()
    out["pulado"] = False
    return out
