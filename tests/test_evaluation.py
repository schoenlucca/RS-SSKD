"""Testes da avaliação: previsões do Mask R-CNN -> copas -> métricas."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rsskd.evaluation import (  # noqa: E402
    CrownMetrics, evaluate_student, predictions_to_instances,
)

H = W = 32


def _saida(prob_masks, scores):
    """Imita a saída do Mask R-CNN no modo de previsão."""
    m = torch.tensor(np.stack(prob_masks), dtype=torch.float32)[:, None]
    return {"masks": m, "scores": torch.tensor(scores),
            "boxes": torch.zeros(len(scores), 4), "labels": torch.ones(len(scores))}


def _quadrado(y0, y1, x0, x1, p=0.9):
    a = np.zeros((H, W))
    a[y0:y1, x0:x1] = p
    return a


def test_sobreposicao_fica_com_a_copa_mais_confiavel():
    a = _quadrado(0, 10, 0, 10)
    b = _quadrado(5, 15, 5, 15)                      # sobrepõe 'a' em 5x5
    inst = predictions_to_instances(_saida([a, b], [0.6, 0.9]))
    assert inst[7, 7] == 1                            # 'b' (0,9) é a copa 1
    assert (inst == 1).sum() == 100                   # 'b' inteira
    assert (inst == 2).sum() == 100 - 25              # 'a' sem a parte de 'b'


def test_limiares_de_confianca_mascara_e_area():
    a = _quadrado(0, 10, 0, 10, p=0.9)
    b = _quadrado(20, 30, 20, 30, p=0.4)              # máscara fraca
    c = _quadrado(0, 3, 20, 23, p=0.9)                # copa de 9 pixels
    d = _quadrado(12, 18, 0, 6, p=0.9)
    inst = predictions_to_instances(_saida([a, b, c, d], [0.9, 0.9, 0.9, 0.3]),
                                    score_thr=0.5, mask_thr=0.5, min_area=10)
    assert inst.max() == 1                            # só 'a' sobra
    assert predictions_to_instances(_saida([a], [0.2])).max() == 0
    vazia = {"masks": torch.zeros(0, 1, H, W), "scores": torch.zeros(0)}
    assert predictions_to_instances(vazia).shape == (H, W)


def test_metricas_somam_todas_as_imagens():
    ref = np.zeros((H, W), np.int32)
    ref[0:10, 0:10], ref[20:30, 20:30] = 1, 2
    perfeita = ref.copy()
    so_uma = np.where(ref == 1, 1, 0).astype(np.int32)
    m = CrownMetrics()
    m.add(perfeita, ref)                              # 2 acertos
    m.add(so_uma, ref)                                # 1 acerto, 1 copa perdida
    s = m.summary()
    assert (s["precisao"], s["revocacao"]) == (1.0, 0.75)
    assert s["f1"] == pytest.approx(2 * 0.75 / 1.75)
    assert s["erro_contagem"] == 0.5 and s["iou_copa"] == 1.0


def test_avaliar_aluno_que_acerta_tudo():
    # Um "aluno" de mentira que devolve as copas da referência: F1 = 1.
    refs = []
    for k in range(3):
        r = np.zeros((H, W), np.int32)
        r[2:12, 2 + 5 * k:12 + 5 * k] = 1
        r[18:28, 10:20] = 2
        refs.append(r)

    class Conjunto(torch.utils.data.Dataset):
        transform = None

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return torch.zeros(6, H, W), {"image_id": torch.tensor(i)}

        def reference(self, i):
            return refs[i]

    class Perfeito(torch.nn.Module):
        def forward(self, images):
            outs = []
            for _ in images:
                i = len(self.vistos)
                self.vistos.append(i)
                r = refs[i]
                masks = [(r == c).astype(float) for c in (1, 2)]
                outs.append(_saida(masks, [0.9, 0.8]))
            return outs

    aluno = Perfeito()
    aluno.vistos = []
    aluno.train()
    resumo, metricas = evaluate_student(aluno, Conjunto(), batch_size=2)
    assert resumo["f1"] == 1.0 and resumo["iou_pixel"] == 1.0
    assert [r["indice"] for r in metricas.rows] == [0, 1, 2]
    assert aluno.training                             # volta ao modo de treino


def test_metricas_de_area():
    from rsskd.evaluation import area_kl

    ref = np.zeros((H, W), np.int32)
    ref[0:10, 0:10] = 1                       # 100 px
    ref[20:30, 20:30] = 2                     # 100 px
    pred = np.zeros((H, W), np.int32)
    pred[0:10, 0:12] = 1                      # 120 px, casa com a copa 1
    pred[20:30, 20:30] = 2                    # 100 px, casa com a copa 2
    pred[12:18, 0:6] = 3                      # 36 px, falso positivo
    m = CrownMetrics()
    m.add(pred, ref)
    s = m.summary()
    # Erros de área nos acertos: 20 e 0 pixels.
    assert s["area_mae"] == pytest.approx(10)
    assert s["area_rmse"] == pytest.approx(np.sqrt((20 ** 2 + 0) / 2))
    assert s["area_mediana_ref"] == 100 and s["area_mediana_pred"] == 100
    assert m.rows[0]["erro_area_medio"] == pytest.approx(10)
    assert s["area_erro_rel"] == pytest.approx((20 / 100 + 0) / 2)
    # KL: zero para distribuições iguais, positiva para diferentes, e maior
    # quanto mais diferentes.
    assert area_kl([100, 200, 400], [100, 200, 400]) == pytest.approx(0, abs=1e-6)
    a = area_kl([100] * 10 + [200] * 10, [100] * 10 + [800] * 10)
    b = area_kl([100] * 10 + [200] * 10, [800] * 20)
    assert 0 < a < b
    assert area_kl([], [100]) == 0.0


def test_erro_de_area_nas_copas_que_todos_acertaram():
    from rsskd.evaluation import common_area_error

    ref = np.zeros((64, 64), np.int32)
    ref[0:10, 0:10] = 1                       # copa pequena, 100 px
    ref[20:40, 20:40] = 2                     # copa grande, 400 px
    # A acerta só a pequena, com 10% de erro.
    a = np.zeros((64, 64), np.int32)
    a[0:10, 0:11] = 1
    # B acerta as duas: a pequena com 10% de erro e a grande com 20%.
    b = np.zeros((64, 64), np.int32)
    b[0:10, 0:11] = 1
    b[20:40, 20:44] = 2
    ma, mb = CrownMetrics(), CrownMetrics()
    ma.add(a, ref, index=0)
    mb.add(b, ref, index=0)
    # Cada um no seu conjunto: B parece pior em pixels (acertou a copa grande).
    assert mb.summary()["area_mae"] > ma.summary()["area_mae"]
    # Nas copas em comum (só a pequena), os dois erram igual.
    c = common_area_error({"A": ma, "B": mb})
    assert c["n_copas"] == 1
    assert c["A"] == pytest.approx(0.1) and c["B"] == pytest.approx(0.1)
