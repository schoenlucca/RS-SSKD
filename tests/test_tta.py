"""Previsão com as 8 orientações combinadas (rsskd/tta.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsskd.augment import DIHEDRAL  # noqa: E402
from rsskd.evaluation import predictions_to_instances  # noqa: E402
from rsskd.tta import merge_detections, predict_tta  # noqa: E402


class CopiaCanal:
    """'Modelo' que prevê uma copa onde o canal 0 é alto, com score 0,9, e,
    se `ruido`, uma copa espúria fixa no canto (que não gira com a imagem)."""

    def __init__(self, ruido: bool = False):
        self.ruido = ruido

    def __call__(self, imgs):
        outs = []
        for x in imgs:
            masks = [(x[0] > 0.5).float()]
            scores = [0.9]
            if self.ruido:
                m = torch.zeros_like(x[0])
                m[:6, :6] = 1.0                          # sempre no mesmo canto
                masks.append(m)
                scores.append(0.95)
            outs.append({"masks": torch.stack(masks)[:, None], "scores": torch.tensor(scores)})
        return outs


def _imagem():
    img = torch.zeros(3, 40, 40)
    img[0, 10:22, 24:34] = 1.0                           # copa fora do centro
    return img


def test_identidade_igual_a_previsao_comum():
    img = _imagem()
    comum = predictions_to_instances(CopiaCanal()([img])[0])
    tta = predict_tta(CopiaCanal(), img, transforms=[(0, False)], min_votes=1)
    assert (predictions_to_instances(tta) == comum).all()


def test_copa_que_gira_com_a_imagem_aparece_nas_8():
    img = _imagem()
    out = predict_tta(CopiaCanal(), img)
    assert out["votes"].tolist() == [8]
    lab = predictions_to_instances(out)
    assert lab.max() == 1 and (lab > 0).sum() == 12 * 10


def test_ruido_que_nao_gira_e_descartado():
    # A copa espúria fica sempre no canto da imagem *transformada*: desfeita a
    # transformação, cai em cantos diferentes, e não junta votos.
    out = predict_tta(CopiaCanal(ruido=True), _imagem())
    assert out["votes"].tolist() == [8]                  # só a copa de verdade
    sem_tta = predictions_to_instances(CopiaCanal(ruido=True)([_imagem()])[0])
    assert sem_tta.max() == 2                            # sem TTA, o ruído entra


def test_votos_e_media_das_mascaras():
    m = torch.zeros(1, 1, 10, 10)
    m[0, 0, 2:8, 2:8] = 1.0
    meio = m * 0.6
    vistas = [{"scores": torch.tensor([0.9]), "masks": m},
              {"scores": torch.tensor([0.7]), "masks": meio},
              {"scores": torch.zeros(0), "masks": torch.zeros(0, 1, 10, 10)},
              {"scores": torch.zeros(0), "masks": torch.zeros(0, 1, 10, 10)}]
    out = merge_detections(vistas, 4)                    # min_votes = 2
    assert out["votes"].tolist() == [2]
    assert out["scores"].item() == pytest.approx(0.8)
    assert out["masks"][0, 0, 5, 5].item() == pytest.approx(0.8)
    assert merge_detections(vistas, 4, min_votes=3)["scores"].numel() == 0
    assert len(DIHEDRAL) == 8


def test_roda_com_o_aluno_de_verdade():
    from rsskd.detector import build_student

    torch.manual_seed(0)
    m = build_student(pretrained_backbone=False, image_size=64).eval()
    out = predict_tta(m, torch.randn(6, 64, 64), score_thr=0.0)
    assert out["masks"].shape[1:] == (1, 64, 64)
    assert predictions_to_instances(out, 0.0).shape == (64, 64)
