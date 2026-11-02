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
"""

from __future__ import annotations

import math

import torch

from .consistency import capture_objectness, choose_transforms, consistency_loss


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
               consistency_weight: float = 1.0) -> dict:
    """Um passo de treino. Devolve as perdas (números) e a perda total.

    consistency:        None (desligada), "artigo" (as 3 transformações do
                        RS-SSKD em todo passo), "sorteio" (uma do D4) ou uma
                        lista de nomes; ver choose_transforms.
    consistency_weight: o lambda que multiplica a consistência (1 no artigo).

    Além das 5 perdas do Mask R-CNN, devolve "loss_consistencia" (sem o
    lambda) e "cons_<transformação>" para cada transformação usada.

    Se alguma perda virar infinita ou NaN (sinal de treino instável), o passo
    é pulado em vez de estragar os pesos, e "pulado" vem True.
    """
    model.train()
    images, targets = to_device(images, targets, device)
    if consistency:
        with capture_objectness(model) as cap:
            losses = model(images, targets)        # perda do professor TEBAKA
    else:
        cap, losses = {}, model(images, targets)
    total = torch.stack(list(losses.values())).sum()
    out = {k: float(v.detach()) for k, v in losses.items()}

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
