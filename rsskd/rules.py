"""Regras de conhecimento de Harmon et al. (2022), etapa 2 (parte 6b).

Harmon et al. (IEEE TGRS 60:4415419) injetam conhecimento ecológico num
detector de copas com uma rede professor-aluno em que **o professor é o
próprio aluno corrigido pelas regras** (regularização posterior, Hu et al.,
2016). Para cada previsão, com probabilidade p(y) de ser copa (y = +) ou
fundo (y = -), o professor é

    q*(y)  proporcional a  p(y) · exp( -C · lambda · (1 - r(y)) )      (eq. 14)

em que r(y) em [0, 1] diz o quanto a regra é satisfeita se a resposta for y.
Uma previsão que viola a regra fica menos provável em q*; o aluno é então
puxado na direção de q* por uma perda de destilação (eq. 16):

    L = (1 - mu) · L_classificação + L_caixas + mu · L_D(q*, p)

**Regra 1 (área média), eq. 33 e 34.** No texto do artigo: "para todas as
previsões, se a área da copa prevista for menor que a área média da área de
estudo, provavelmente é uma árvore". Ela usa uma sigmoide invertida da área
da caixa prevista,

    bboxSize(A) = 1 / (1 + exp(-k · (mu_area - A)))

que vale ~1 para caixas bem menores que a média, 0,5 na média e ~0 para as
bem maiores. Com isso, r(+) = bboxSize(A) e r(-) = 1 - bboxSize(A): uma
caixa grande demais tem a probabilidade de copa reduzida. A regra é de um
lado só: ela pune copas **grandes** (no nosso caso, as copas fundidas do
TEBAKA), e não pune copas pequenas. Harmon justifica: a distribuição de
tamanhos é assimétrica (a maioria das árvores é menor que a média) e o
detector tende a superestimar o tamanho.

Adaptações ao nosso caso:
  - mu_area vem das **copas do professor TEBAKA** no treino (não há rótulos
    humanos); por padrão a mediana, e não a média, porque as copas fundidas
    do TEBAKA puxam a média para cima (a média fica como opção, fiel ao
    artigo);
  - a área é a da caixa, como no artigo, comparada com as caixas das copas
    do TEBAKA (caixa com caixa);
  - k é dado em relação a mu_area (a sigmoide usa (mu_area - A) / mu_area),
    para o mesmo k servir em imagens de outra resolução: com k = 5, uma
    caixa 50% maior que a referência fica com bboxSize ~ 0,08.
  - a regra age na cabeça de classificação do Mask R-CNN, nas mesmas regiões
    em que a auto-destilação age (capturadas por distill.capture_roi), como
    no artigo, em que as regras 1, 3 e 4 agem na cabeça de classificação.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def bbox_size(area: torch.Tensor, mu_area: float, k: float = 5.0) -> torch.Tensor:
    """bboxSize(A) de Harmon (eq. 33), com k relativo a mu_area."""
    return torch.sigmoid(k * (mu_area - area) / mu_area)


def box_areas(boxes) -> torch.Tensor:
    """Áreas (pixels) das caixas [x1, y1, x2, y2]; aceita uma lista por imagem."""
    if isinstance(boxes, (list, tuple)):
        boxes = torch.cat(list(boxes)) if len(boxes) else torch.zeros(0, 4)
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


@torch.no_grad()
def rule_teacher(class_logits: torch.Tensor, areas: torch.Tensor, mu_area: float,
                 k: float = 5.0, C: float = 1.0, lam: float = 1.0) -> torch.Tensor:
    """O professor q* da equação 14 para a regra 1, sem gradiente.

    class_logits: (n, 2) notas do aluno (classe 0 = fundo, 1 = copa).
    areas:        (n,) áreas das caixas dessas regiões.
    Devolve q* (n, 2), cada linha somando 1.
    """
    s = bbox_size(areas, mu_area, k)                       # r(+); r(-) = 1 - s
    log_p = F.log_softmax(class_logits, dim=1)
    penalty = C * lam * torch.stack([s, 1 - s], dim=1)     # 1 - r(y): fundo, copa
    return F.softmax(log_p - penalty, dim=1)


def rule_loss(class_logits: torch.Tensor, areas: torch.Tensor, mu_area: float,
              k: float = 5.0, C: float = 1.0, lam: float = 1.0) -> torch.Tensor:
    """L_D da equação 16: KL(q* || p), puxando o aluno para o professor das regras."""
    if class_logits.numel() == 0:
        return class_logits.sum() * 0.0
    q = rule_teacher(class_logits, areas, mu_area, k, C, lam)
    return F.kl_div(F.log_softmax(class_logits, dim=1), q, reduction="batchmean")


def rules_loss_from_capture(cap: dict, mu_area: float, k: float = 5.0,
                            C: float = 1.0, lam: float = 1.0) -> tuple[torch.Tensor, dict]:
    """A regra 1 nas regiões que o aluno usou neste passo (distill.capture_roi).

    Devolve (perda, {"regra1": valor, "regra1_satisfeita": média de r(+)
    nas regiões que o aluno acha mais provável serem copa}).
    """
    logits = cap["class_logits"]
    areas = box_areas(cap["proposals"]).to(logits.device)
    loss = rule_loss(logits, areas, mu_area, k, C, lam)
    with torch.no_grad():
        crown = logits.argmax(1) == 1
        sat = bbox_size(areas[crown], mu_area, k).mean() if crown.any() else torch.tensor(1.0)
    return loss, {"regra1": float(loss.detach()), "regra1_satisfeita": float(sat)}


def check_area_stat(stat: str) -> str:
    """Confere a estatística da área de referência: "mediana", "media" ou
    "p<N>", um percentil entre 1 e 99 (ex.: "p90"). Devolve a mesma string."""
    if stat in ("mediana", "media"):
        return stat
    if stat.startswith("p") and stat[1:].isdigit() and 1 <= int(stat[1:]) <= 99:
        return stat
    raise ValueError(f"Estatística de área desconhecida: {stat!r}. "
                     "Use mediana, media ou p<N> (ex.: p90).")


def reference_area(labels_iter, stat: str = "mediana") -> float:
    """Área de referência mu_area das caixas das copas, numa sequência de
    imagens de etiquetas (0 = fundo).

    stat: "mediana" (padrão), "media" (a de Harmon) ou "p<N>", o percentil N
    (ex.: "p90"). Com a mediana, metade das copas do próprio TEBAKA fica
    acima de mu_area por construção, e a regra pesa sobre copas normais;
    um percentil alto põe o limite perto das copas grandes demais (em geral,
    árvores vizinhas que o professor fundiu), que é o alvo da regra.

    Uso, com as copas do TEBAKA no treino:
        store = PseudoLabelStore("data/pseudo/agml")
        mu = reference_area(store.load("train", i) for i in store.indices("train"))
        mu90 = reference_area((store.load("train", i) for i in ...), stat="p90")
    """
    check_area_stat(stat)
    areas = []
    for labels in labels_iter:
        for sl in ndi.find_objects(labels):
            if sl is not None:
                areas.append((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))
    if not areas:
        raise ValueError("Nenhuma copa para calcular a área de referência.")
    if stat == "mediana":
        return float(np.median(areas))
    if stat == "media":
        return float(np.mean(areas))
    return float(np.percentile(areas, int(stat[1:])))
