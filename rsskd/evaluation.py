"""Avaliação: transforma a saída do aluno em copas e mede contra a referência.

É a segunda peça do item 3.

**De previsões a copas.** O Mask R-CNN devolve, para cada imagem, uma lista
de detecções, cada uma com:

    score:  confiança de que é uma copa (0 a 1);
    box:    a caixa;
    mask:   um mapa de probabilidade (H, W), "quanto este pixel pertence a
            esta copa".

As máscaras de detecções diferentes podem se sobrepor (o modelo não proíbe
um pixel de pertencer a duas copas). Para comparar com a referência, que é
uma imagem de etiquetas (0 = fundo, 1..N = copas), cada pixel tem de ficar
com uma copa só. `predictions_to_instances` faz isso assim:

    1. descarta detecções com score < score_thr (pouco confiáveis);
    2. binariza cada máscara: pixel é da copa se a probabilidade >= mask_thr;
    3. pinta as copas da mais confiável para a menos confiável; um pixel já
       pintado não é tomado por uma copa menos confiável;
    4. descarta copas que ficaram com menos de min_area pixels.

**As métricas** são as mesmas usadas no professor (Harmon et al., 2022),
agora num lugar só (`CrownMetrics`), para que professor e aluno sejam
medidos exatamente do mesmo jeito:

    precisão, revocação e F1 por copa (acerto = IoU >= 0,5), somando
    acertos e erros de todas as imagens antes de dividir (média "micro");
    IoU médio das copas acertadas; IoU por pixel (copa contra fundo);
    erro médio de contagem (|copas previstas - copas reais| por imagem);
    e as métricas de área de Harmon et al. (parte 7a):
      - RMSE e MAE da área (pixels), entre cada copa acertada e a copa da
        referência com que ela casou: o contorno está do tamanho certo?
      - KL da distribuição de áreas: compara o histograma das áreas de
        todas as copas previstas com o das copas da referência, acertadas
        ou não. É a métrica que mostra se a regra 1 corrige as copas
        fundidas (grandes demais), mesmo quando o F1 pouco muda.
"""

from __future__ import annotations

import numpy as np
import torch

from .augment import make_loader
from .data import iou_binary, match_instances


# --------------------------------------------------------------------------
# Previsões do Mask R-CNN -> imagem de copas
# --------------------------------------------------------------------------
def predictions_to_instances(output: dict, score_thr: float = 0.5,
                             mask_thr: float = 0.5, min_area: int = 0) -> np.ndarray:
    """Converte a saída do Mask R-CNN de uma imagem em etiquetas (H, W).

    output: {"scores": (n,), "masks": (n, 1, H, W) probabilidades, ...},
            como o modelo devolve no modo de previsão (model.eval()).
    Retorna int32 (H, W): 0 = fundo, 1..N = copas, numeradas da mais
    confiável para a menos confiável.
    """
    masks = output["masks"].detach().cpu()
    scores = output["scores"].detach().cpu()
    h, w = masks.shape[-2:]
    labels = np.zeros((h, w), np.int32)

    order = torch.argsort(scores, descending=True)
    next_id = 1
    for k in order.tolist():
        if scores[k] < score_thr:
            break                                   # o resto é ainda menor
        crown = (masks[k, 0] >= mask_thr).numpy() & (labels == 0)
        if crown.sum() < max(min_area, 1):
            continue
        labels[crown] = next_id
        next_id += 1
    return labels


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
class CrownMetrics:
    """Acumula as métricas imagem por imagem e resume no fim.

    Uso:
        m = CrownMetrics()
        for pred, ref in ...:
            m.add(pred, ref)            # imagens de etiquetas (H, W)
        m.summary()                     # {"precisao": ..., "f1": ..., ...}
        m.rows                          # uma linha por imagem (para CSV)
    """

    def __init__(self, iou_thr: float = 0.5):
        self.iou_thr = iou_thr
        self.rows: list[dict] = []
        self.pred_areas: list[int] = []        # todas as copas previstas
        self.ref_areas: list[int] = []         # todas as copas da referência
        self.matched_areas: list[tuple[int, int]] = []   # (prevista, referência)

    def add(self, pred: np.ndarray, ref: np.ndarray, index: int | None = None,
            pred_mask: np.ndarray | None = None) -> dict:
        """Compara uma imagem. pred_mask: máscara copa/fundo para o IoU por
        pixel, se for diferente de pred > 0 (o professor usa a máscara de
        vegetação antes da separação em copas)."""
        m = match_instances(pred, ref, self.iou_thr)
        row = {"indice": index, "n_ref": m["n_ref"], "n_pred": m["n_pred"],
               "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
               "iou_copa": m["mean_iou"],
               "iou_pixel": iou_binary(pred > 0 if pred_mask is None else pred_mask,
                                       ref > 0)}
        p_area = np.bincount(pred.ravel().astype(np.int64))
        r_area = np.bincount(ref.ravel().astype(np.int64))
        self.pred_areas += [int(a) for a in p_area[1:] if a > 0]
        self.ref_areas += [int(a) for a in r_area[1:] if a > 0]
        pairs = [(int(p_area[p]), int(r_area[r])) for p, r, _ in m["pairs"]]
        self.matched_areas += pairs
        row["erro_area_medio"] = (float(np.mean([abs(a - b) for a, b in pairs]))
                                  if pairs else "")
        self.rows.append(row)
        return row

    def summary(self) -> dict:
        tp = sum(r["tp"] for r in self.rows)
        fp = sum(r["fp"] for r in self.rows)
        fn = sum(r["fn"] for r in self.rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        matched = [r["iou_copa"] for r in self.rows if r["tp"]]
        return {
            "precisao": prec, "revocacao": rec, "f1": f1,
            "iou_copa": float(np.mean(matched)) if matched else 0.0,
            "iou_pixel": float(np.mean([r["iou_pixel"] for r in self.rows]))
                         if self.rows else 0.0,
            "erro_contagem": float(np.mean([abs(r["n_pred"] - r["n_ref"])
                                            for r in self.rows])) if self.rows else 0.0,
        } | self.area_summary()

    def area_summary(self) -> dict:
        """Métricas de área de Harmon et al. (2022): RMSE, MAE e KL."""
        if self.matched_areas:
            d = np.array([a - b for a, b in self.matched_areas], dtype=float)
            rmse, mae = float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))
        else:
            rmse = mae = 0.0
        return {"area_rmse": rmse, "area_mae": mae,
                "area_kl": area_kl(self.ref_areas, self.pred_areas),
                "area_mediana_pred": float(np.median(self.pred_areas)) if self.pred_areas else 0.0,
                "area_mediana_ref": float(np.median(self.ref_areas)) if self.ref_areas else 0.0}


def area_kl(ref_areas, pred_areas, bins: int = 20, eps: float = 0.5) -> float:
    """KL(referência || previsto) entre as distribuições de áreas das copas.

    As áreas vão para os mesmos `bins` intervalos, em escala logarítmica
    (áreas de copas variam por ordens de grandeza), entre a menor e a maior
    área das duas listas. Cada histograma vira uma distribuição de
    probabilidade; antes, soma-se eps = 0,5 copa a cada intervalo, para um
    intervalo vazio não fazer a KL explodir por causa de uma copa só. Vale 0
    quando as duas distribuições coincidem e cresce quando diferem; não é
    simétrica: mede o quanto o previsto falha em reproduzir a referência.
    """
    if not ref_areas or not pred_areas:
        return 0.0
    allv = np.asarray(list(ref_areas) + list(pred_areas), dtype=float)
    lo, hi = max(allv.min(), 1.0), allv.max()
    if hi <= lo:
        return 0.0
    edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    p = np.histogram(ref_areas, edges)[0] + eps
    q = np.histogram(pred_areas, edges)[0] + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def format_summary(s: dict) -> str:
    """Resumo numa linha, para imprimir."""
    return (f"P={s['precisao']:.2f} R={s['revocacao']:.2f} F1={s['f1']:.2f} "
            f"IoU_copa={s['iou_copa']:.2f} IoU_pixel={s['iou_pixel']:.2f} "
            f"erro_contagem={s['erro_contagem']:.1f} "
            f"MAE_area={s.get('area_mae', 0):.0f}px KL_area={s.get('area_kl', 0):.2f}")


# --------------------------------------------------------------------------
# Avaliação do aluno
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_student(model, dataset, device: str | torch.device = "cpu",
                     score_thr: float = 0.5, mask_thr: float = 0.5,
                     min_area: int = 0, batch_size: int = 4,
                     teacher_metrics: CrownMetrics | None = None):
    """Roda o aluno em todas as imagens de `dataset` e mede contra a referência.

    dataset: um CrownDataset **sem** transformação (as imagens na orientação
             original, para coincidirem com a referência).
    Retorna (resumo, CrownMetrics). As copas do professor que vêm no alvo
    são ignoradas: o aluno é medido contra a referência.

    teacher_metrics: se for passado um CrownMetrics, ele é preenchido, na
    mesma passada, com a comparação do aluno contra as copas do professor
    (TEBAKA) nessas imagens. Não usa nenhum rótulo humano: é o critério
    "sem rótulos" para escolher o melhor modelo (parte 5b).
    """
    if getattr(dataset, "transform", None) is not None:
        raise ValueError("Avalie com um CrownDataset sem transform.")
    was_training = model.training
    model.eval()
    metrics = CrownMetrics()
    for images, targets in make_loader(dataset, batch_size=batch_size, shuffle=False):
        outputs = model([im.to(device) for im in images])
        for out, t in zip(outputs, targets):
            i = int(t["image_id"])
            pred = predictions_to_instances(out, score_thr, mask_thr, min_area)
            metrics.add(pred, dataset.reference(i), index=i)
            if teacher_metrics is not None:
                teacher_metrics.add(pred, dataset.store.load(dataset.split, i), index=i)
    model.train(was_training)
    return metrics.summary(), metrics
