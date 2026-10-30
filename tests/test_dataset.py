"""Testes do conjunto de dados do aluno (imagem + copas do professor)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from rsskd.data import AgmlTreeCrowns  # noqa: E402
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.inputs import compute_input_spec  # noqa: E402
from rsskd.pseudolabels import PseudoLabelStore  # noqa: E402
from test_data import NIR, RED, H, W, _fake_row  # noqa: E402


@pytest.fixture()
def pastas(tmp_path):
    """Conjunto falso (3 treino, 1 validação) e copas do professor gravadas.

    As copas do professor são escolhidas à mão, diferentes da referência,
    para conferir que o alvo vem do professor e não da referência.
    """
    dados = tmp_path / "dados"
    dados.mkdir()
    rows = [_fake_row(i, s) for i, s in enumerate(["train"] * 3 + ["val"])]
    pd.DataFrame(rows).to_parquet(dados / "train-00000.parquet")

    store = PseudoLabelStore(tmp_path / "pseudo")
    for i in range(3):
        copas = np.zeros((H, W), np.int32)
        for k in range(i + 1):                      # imagem i tem i+1 copas
            copas[5 + 15 * k:15 + 15 * k, 5:20] = k + 1
        store.save("train", i, copas)
    store.save("val", 0, np.zeros((H, W), np.int32))   # sem nenhuma copa

    imgs = AgmlTreeCrowns(dados, split="train")
    nomes = ["b0", "b1", "b2", "b3", "b4", "ndvi"]
    spec = compute_input_spec((imgs[i].bands for i in range(len(imgs))),
                              NIR, RED, nomes)
    return dados, tmp_path / "pseudo", spec


def test_formato_do_mask_rcnn(pastas):
    dados, pseudo, spec = pastas
    ds = CrownDataset("train", pseudo, spec, dataset_dir=dados)
    assert len(ds) == 3
    imagem, alvo = ds[2]
    assert imagem.shape == (6, H, W) and imagem.dtype == torch.float32
    assert alvo["boxes"].shape == (3, 4) and alvo["boxes"].dtype == torch.float32
    assert alvo["labels"].tolist() == [1, 1, 1]
    assert alvo["masks"].shape == (3, H, W) and alvo["masks"].dtype == torch.uint8
    # Caixa da 1a copa: colunas 5..19 e linhas 5..14 -> [5, 5, 20, 15].
    assert alvo["boxes"][0].tolist() == [5, 5, 20, 15]
    assert int(alvo["image_id"]) == 2


def test_alvo_vem_do_professor_e_nao_da_referencia(pastas):
    dados, pseudo, spec = pastas
    ds = CrownDataset("train", pseudo, spec, dataset_dir=dados)
    _, alvo = ds[0]
    ref = ds.reference(0)
    assert alvo["masks"].shape[0] == 1            # professor: 1 copa
    assert ref.max() == 2                         # referência: 2 copas
    assert not np.array_equal(alvo["masks"][0].numpy(), ref > 0)


def test_imagem_sem_copas(pastas):
    dados, pseudo, spec = pastas
    _, alvo = CrownDataset("val", pseudo, spec, dataset_dir=dados)[0]
    assert alvo["boxes"].shape == (0, 4)
    assert alvo["masks"].shape == (0, H, W)


def test_transformacao_aplicada_nas_duas_partes(pastas):
    dados, pseudo, spec = pastas

    def espelha(bandas, copas):                    # espelhamento horizontal
        return bandas[:, :, ::-1], copas[:, ::-1]

    normal = CrownDataset("train", pseudo, spec, dataset_dir=dados)
    espelhado = CrownDataset("train", pseudo, spec, dataset_dir=dados,
                             transform=espelha)
    (x, a), (xe, ae) = normal[0], espelhado[0]
    assert torch.equal(xe, x.flip(-1))
    assert torch.equal(ae["masks"], a["masks"].flip(-1))
    # Caixa espelhada: x_min' = W - x_max e x_max' = W - x_min.
    assert ae["boxes"][0].tolist() == [W - 20, 5, W - 5, 15]


def test_erro_se_faltam_copas(pastas, tmp_path):
    dados, _, spec = pastas
    with pytest.raises(FileNotFoundError, match="sem copas"):
        CrownDataset("train", tmp_path / "vazia", spec, dataset_dir=dados)
