"""Auto-destilação do RS-SSKD, adaptada ao Mask R-CNN (etapa 2, parte 6a).

No RS-SSKD (Zhang et al., 2021, equação 6), a etapa 2 treina um aluno novo
com a ajuda do modelo da etapa 1, congelado, como professor:

    L = CE(aluno, rótulos) + KL( aluno/T || professor/T ) * T^2

A KL faz o aluno imitar as probabilidades *suaves* do professor (por
exemplo, "70% copa, 30% fundo"), e não só a resposta dura. A temperatura T
suaviza ainda mais essas probabilidades; multiplicar por T^2 mantém a
escala dos gradientes independente de T (Hinton et al., 2015).

**Onde aplicar a KL no Mask R-CNN (opção b, escolhida).** O Mask R-CNN não
dá um vetor de probabilidades por imagem, e sim por região proposta. Então:

    1. o aluno roda normalmente; ganchos guardam as regiões que ele usou
       e as saídas das suas cabeças nessas regiões:
         - notas de classificação (fundo, copa), uma por região;
         - máscaras 28x28 (notas por pixel), uma por região positiva;
    2. o professor roda o backbone dele e as cabeças dele **nas mesmas
       regiões** do aluno, sem gradiente;
    3. KL com temperatura entre os dois, região a região:
         - classificação: KL entre as distribuições de 2 classes, exatamente
           a equação 6, só que por região;
         - máscaras: KL entre as probabilidades de cada pixel (copa ou não),
           calculada pixel a pixel e tirada a média.

Usar as mesmas regiões é essencial: se cada rede propusesse as suas, as
listas seriam diferentes e não haveria o que comparar.

A KL das máscaras é a que pode melhorar o contorno: onde o TEBAKA marcou uma
borda serrilhada, o professor da etapa 1 tende a dar probabilidades
intermediárias, e o aluno aprende essa borda suave.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Cópia congelada do modelo: modo de avaliação e pesos sem gradiente.

    É o "professor" da auto-destilação: os pesos dele nunca mudam.
    """
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@contextmanager
def capture_roi(model):
    """Guarda, durante um passo de treino do aluno, o que a destilação precisa.

    Uso:
        with capture_roi(aluno) as cap:
            perdas = aluno(imagens, alvos)      # passo normal (TEBAKA)
        cap["images"]          lote (B, C, H, W) que entrou no backbone
        cap["image_shapes"]    tamanhos das imagens
        cap["proposals"]       regiões usadas na cabeça de classificação
        cap["class_logits"]    notas (n_regiões, 2) do aluno, com gradiente
        cap["mask_proposals"]  regiões positivas, usadas na cabeça de máscaras
        cap["mask_logits"]     notas (n_positivas, 2, 28, 28) do aluno

    Funciona com ganchos ("hooks"), como capture_objectness na consistência:
    nada no Mask R-CNN do torchvision precisa ser modificado.
    """
    cap: dict = {}
    roi = model.roi_heads

    def on_backbone(_m, args):
        cap["images"] = args[0]

    def on_box_pool(_m, args):
        cap["proposals"], cap["image_shapes"] = args[1], args[2]

    def on_box_predictor(_m, _args, output):
        cap["class_logits"] = output[0]

    def on_mask_pool(_m, args):
        cap["mask_proposals"] = args[1]

    def on_mask_predictor(_m, _args, output):
        cap["mask_logits"] = output

    handles = [
        model.backbone.register_forward_pre_hook(on_backbone),
        roi.box_roi_pool.register_forward_pre_hook(on_box_pool),
        roi.box_predictor.register_forward_hook(on_box_predictor),
        roi.mask_roi_pool.register_forward_pre_hook(on_mask_pool),
        roi.mask_predictor.register_forward_hook(on_mask_predictor),
    ]
    try:
        yield cap
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def teacher_roi_outputs(teacher, cap: dict):
    """Notas do professor nas mesmas regiões do aluno (sem gradiente).

    Devolve (class_logits (n, 2), mask_logits (n_pos, 2, 28, 28)).
    """
    roi = teacher.roi_heads
    feats = teacher.backbone(cap["images"])
    box = roi.box_head(roi.box_roi_pool(feats, cap["proposals"], cap["image_shapes"]))
    class_logits, _ = roi.box_predictor(box)
    mask = roi.mask_head(roi.mask_roi_pool(feats, cap["mask_proposals"], cap["image_shapes"]))
    return class_logits, roi.mask_predictor(mask)


def kd_classification(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    """Equação 6 do RS-SSKD, por região: KL(professor/T || aluno/T) * T^2."""
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    T = temperature
    return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * T * T


def kd_masks(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    """KL por pixel entre as máscaras, com temperatura, na média dos pixels.

    Cada pixel é uma distribuição de 2 resultados (copa ou não), com
    probabilidade sigmoid(nota/T). A KL binária é calculada como
        BCE(aluno, alvo = p_professor) - entropia(p_professor),
    que vale zero quando as duas probabilidades coincidem. Usa o canal da
    classe "copa" (1), o mesmo que a perda de máscaras do torchvision usa.
    """
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    T = temperature
    s = student_logits[:, 1] / T
    p_t = torch.sigmoid(teacher_logits[:, 1] / T)
    bce = F.binary_cross_entropy_with_logits(s, p_t)
    entropy = F.binary_cross_entropy(p_t, p_t)        # constante para o aluno
    return (bce - entropy) * T * T


def distillation_loss(cap: dict, teacher, temperature: float = 4.0,
                      mask_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Perda de auto-destilação do passo: KL na classificação + KL nas máscaras.

    cap:         o que capture_roi guardou no passo do aluno.
    teacher:     o aluno da etapa 1, congelado (freeze).
    temperature: T da equação 6 (4, como no nosso distill_loss original).
    mask_weight: peso da parte das máscaras (1 = mesmo peso da classificação).
    """
    t_cls, t_mask = teacher_roi_outputs(teacher, cap)
    cls = kd_classification(cap["class_logits"], t_cls, temperature)
    mask = kd_masks(cap["mask_logits"], t_mask, temperature)
    total = cls + mask_weight * mask
    return total, {"kd_cls": float(cls.detach()), "kd_mask": float(mask.detach())}
