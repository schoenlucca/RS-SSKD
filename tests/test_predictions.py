"""Testes das previsões em arquivo (parte 7c)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
Image = pytest.importorskip("PIL.Image")

from rsskd.data import AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.predictions import PredictionStore, check_against, split_manifest  # noqa: E402
from test_data import _fake_row  # noqa: E402


@pytest.fixture()
def conjunto(tmp_path):
    dados = tmp_path / "dados"
    dados.mkdir()
    rows = [_fake_row(i, "test") for i in range(3)]
    pd.DataFrame(rows).to_parquet(dados / "train-00000.parquet")
    return dados, AgmlTreeCrowns(dados, split="test")


def test_png_e_npz_e_conferencia(conjunto, tmp_path):
    _, ds = conjunto
    store = PredictionStore(tmp_path / "prev")
    ref0 = mask_to_instances(ds[0].mask)
    store.save("test", 0, ref0)                                   # .npz
    store.save("test", 1, ref0)
    (tmp_path / "prev" / "test").mkdir(exist_ok=True)
    Image.fromarray(ref0.astype(np.uint16)).save(tmp_path / "prev" / "test" / "0002.png")
    assert store.indices("test") == [0, 1, 2]
    assert np.array_equal(store.load("test", 2), ref0)            # PNG de 16 bits
    assert check_against(store, "test", ds) == []

    # Com a lista das imagens no info.json, uma ordem trocada é detectada.
    manifest = split_manifest(ds)
    manifest[1], manifest[2] = manifest[2], manifest[1]
    store.write_info(metodo="teste", imagens={"test": manifest})
    assert any("ordem trocada" in p for p in check_against(store, "test", ds))


def test_faltando_ou_tamanho_errado(conjunto, tmp_path):
    _, ds = conjunto
    store = PredictionStore(tmp_path / "prev")
    store.save("test", 0, np.zeros((10, 10), np.int32))
    problemas = check_against(store, "test", ds)
    assert any("sem previsão" in p for p in problemas)
    assert any("previsão (10, 10)" in p for p in problemas)


def test_avaliar_pastas(conjunto, tmp_path):
    from evaluate_predictions import main as avaliar

    dados, ds = conjunto
    perfeita, vazia = PredictionStore(tmp_path / "perfeita"), PredictionStore(tmp_path / "vazia")
    for i in range(len(ds)):
        ref = mask_to_instances(ds[i].mask)
        perfeita.save("test", i, ref)
        vazia.save("test", i, np.zeros_like(ref))
    perfeita.write_info(metodo="referência copiada")
    linhas = avaliar([str(tmp_path / "perfeita"), str(tmp_path / "vazia"), "--split", "test",
                      "--pasta", str(dados), "--saida", str(tmp_path / "saidas")])
    assert linhas[0]["previsoes"] == "referência copiada" and linhas[0]["f1"] == 1.0
    assert linhas[1]["previsoes"] == "vazia" and linhas[1]["revocacao"] == 0.0
    md = (tmp_path / "saidas" / "previsoes_test.md").read_text(encoding="utf-8")
    assert "| referência copiada | 1.00 | 1.00 | 1.00 |" in md

    # Uma pasta incompleta interrompe a avaliação, em vez de dar números errados.
    incompleta = PredictionStore(tmp_path / "incompleta")
    incompleta.save("test", 0, mask_to_instances(ds[0].mask))
    with pytest.raises(SystemExit, match="sem previsão"):
        avaliar([str(tmp_path / "incompleta"), "--split", "test", "--pasta", str(dados),
                 "--saida", str(tmp_path / "saidas")])
