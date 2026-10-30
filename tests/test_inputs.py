"""Testes da entrada do aluno (bandas + NDVI, normalizadas)."""

from pathlib import Path

import numpy as np
import pytest

from rsskd.inputs import ChannelStats, InputSpec, build_input, compute_input_spec

NIR, RED = 3, 0


def _imagens(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (5, 32, 32)).astype(np.uint8) for _ in range(n)]


def test_ndvi_e_o_ultimo_canal():
    bands = _imagens(1)[0]
    x = build_input(bands, NIR, RED)
    assert x.shape == (6, 32, 32) and x.dtype == np.float32
    nir, red = bands[NIR].astype(np.float32), bands[RED].astype(np.float32)
    assert np.allclose(x[5], (nir - red) / (nir + red + 1e-6))
    assert np.array_equal(x[:5], bands)                  # bandas intactas
    assert build_input(bands, NIR, RED, add_ndvi=False).shape == (5, 32, 32)


def test_estatisticas_acumuladas_iguais_as_diretas():
    # Somar imagem por imagem tem de dar o mesmo que empilhar tudo.
    xs = [build_input(b, NIR, RED) for b in _imagens()]
    stats = ChannelStats()
    for x in xs:
        stats.update(x)
    tudo = np.concatenate([x.reshape(6, -1) for x in xs], axis=1)
    assert np.allclose(stats.mean, tudo.mean(axis=1))
    assert np.allclose(stats.std, tudo.std(axis=1), rtol=1e-5)


def test_normalizacao_media_zero_desvio_um(tmp_path):
    imgs = _imagens()
    nomes = ["vermelho", "verde", "azul", "nir", "red_edge", "ndvi"]
    spec = compute_input_spec(imgs, NIR, RED, nomes)
    assert spec.in_channels == 6
    xs = np.stack([spec(b) for b in imgs])                # (4, 6, 32, 32)
    assert np.allclose(xs.mean(axis=(0, 2, 3)), 0, atol=1e-4)
    assert np.allclose(xs.std(axis=(0, 2, 3)), 1, atol=1e-3)

    # Gravar e ler de volta dá a mesma receita.
    spec.save(tmp_path / "entrada.json", comment="teste")
    assert InputSpec.load(tmp_path / "entrada.json") == spec

    with pytest.raises(ValueError):                      # nomes errados
        compute_input_spec(imgs, NIR, RED, nomes[:3])


def test_arquivo_do_repositorio():
    cfg = Path(__file__).resolve().parents[1] / "configs" / "agml_entrada.json"
    if not cfg.exists():
        pytest.skip("rode scripts/compute_input_stats.py para gerar o arquivo")
    spec = InputSpec.load(cfg)
    assert spec.in_channels == 6 and spec.channels[-1] == "ndvi"
    assert all(s > 0 for s in spec.std)
