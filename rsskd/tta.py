"""Previsão com as 8 orientações combinadas (TTA, test-time augmentation).

Uma copa de verdade não depende de como a foto foi girada, mas o aluno, sim:
no Cerrado, cerca de 20% das copas mudam quando a imagem é girada ou
espelhada (estabilidade de ~0,80 em scripts/check_without_labels.py). O TTA
usa isso a favor: o aluno prevê a mesma imagem nas 8 orientações do grupo
diedral D4 (4 rotações × com e sem espelho), cada previsão é desfeita de
volta à orientação original, e as copas das 8 são combinadas. O que aparece
em quase todas as orientações fica; o que aparece numa ou duas (geralmente
ruído) sai. Não precisa treinar de novo: vale para qualquer modelo salvo.

Como as copas são combinadas (agrupamento guloso das máscaras):

  1. juntam-se as detecções das 8 previsões com score >= score_thr, cada
     máscara (probabilidades) já de volta à orientação original;
  2. em ordem decrescente de score, cada detecção entra no primeiro grupo
     cuja máscara principal (a do maior score) ela cobre com IoU >= iou_thr,
     desde que o grupo ainda não tenha uma detecção da mesma orientação;
     senão, abre um grupo novo;
  3. um grupo vira copa se reuniu pelo menos `min_votes` orientações (padrão:
     metade delas); a máscara da copa é a média das probabilidades do grupo,
     e o score, a média dos scores.

A saída tem o mesmo formato da saída do Mask R-CNN ({"scores", "masks"}),
então predictions_to_instances e o resto da avaliação não mudam. Com uma
orientação só (a identidade) e min_votes=1, o resultado é o da previsão
comum.

Referências: o TTA por média de aumentos é prática comum desde
KRIZHEVSKY, A.; SUTSKEVER, I.; HINTON, G. ImageNet Classification with Deep
Convolutional Neural Networks. NeurIPS, 2012; a fusão de detecções por
agrupamento segue a ideia da Weighted Boxes Fusion (SOLOVYEV, R. et al.
Image and Vision Computing, 2021), aplicada às máscaras.
"""

from __future__ import annotations

import math

import torch

from .augment import DIHEDRAL, dihedral_torch, inverse_dihedral


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 0.0


def merge_detections(per_view: list[dict], n_views: int, score_thr: float = 0.5,
                     mask_thr: float = 0.5, iou_thr: float = 0.5,
                     min_votes: int | None = None) -> dict:
    """Combina as detecções de várias orientações (já desfeitas).

    per_view: uma saída do Mask R-CNN por orientação, {"scores": (n,),
              "masks": (n, 1, H, W)}, com as máscaras na orientação original.
    Devolve {"scores": (k,), "masks": (k, 1, H, W), "votes": (k,)}.
    """
    if min_votes is None:
        min_votes = math.ceil(n_views / 2)
    dets = []                                     # (score, prob, binário, orientação)
    for v, out in enumerate(per_view):
        for s, m in zip(out["scores"].detach().cpu(), out["masks"].detach().cpu()):
            if s >= score_thr:
                dets.append((float(s), m[0].float(), m[0] >= mask_thr, v))
    dets.sort(key=lambda d: -d[0])

    groups: list[dict] = []
    for s, prob, binary, v in dets:
        for g in groups:
            if v not in g["views"] and _iou(binary, g["main"]) >= iou_thr:
                g["views"].add(v)
                g["scores"].append(s)
                g["probs"].append(prob)
                break
        else:
            groups.append({"main": binary, "views": {v}, "scores": [s], "probs": [prob]})

    kept = [g for g in groups if len(g["views"]) >= min_votes]
    if not kept:
        h, w = per_view[0]["masks"].shape[-2:]
        return {"scores": torch.zeros(0), "masks": torch.zeros(0, 1, h, w),
                "votes": torch.zeros(0, dtype=torch.int64)}
    return {"scores": torch.tensor([sum(g["scores"]) / len(g["scores"]) for g in kept]),
            "masks": torch.stack([torch.stack(g["probs"]).mean(0) for g in kept])[:, None],
            "votes": torch.tensor([len(g["views"]) for g in kept])}


@torch.no_grad()
def predict_tta(model, image: torch.Tensor, transforms=DIHEDRAL, score_thr: float = 0.5,
                mask_thr: float = 0.5, iou_thr: float = 0.5,
                min_votes: int | None = None) -> dict:
    """Prevê `image` (C, H, W) em cada orientação e combina as copas.

    transforms: lista de (k, espelha); padrão: as 8 do D4.
    O modelo deve estar em modo de previsão (model.eval()) e no mesmo
    dispositivo da imagem. As 8 orientações vão num lote só.
    """
    views = [dihedral_torch(image, k, flip) for k, flip in transforms]
    outputs = model(views)
    per_view = []
    for (k, flip), out in zip(transforms, outputs):
        per_view.append({"scores": out["scores"],
                         "masks": inverse_dihedral(out["masks"], k, flip)})
    return merge_detections(per_view, len(transforms), score_thr, mask_thr, iou_thr,
                            min_votes)
