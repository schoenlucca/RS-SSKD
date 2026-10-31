"""Testes dos aumentos de dados (rotações e espelhamentos) e dos lotes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rsskd.augment import (  # noqa: E402
    DIHEDRAL, RandomDihedral, collate, dihedral, dihedral_torch, inverse_dihedral,
    make_loader,
)


def _assimetrica():
    """Imagem sem nenhuma simetria: cada transformação dá um resultado diferente."""
    return np.arange(5 * 6 * 6).reshape(5, 6, 6)


def test_oito_transformacoes_diferentes():
    x = _assimetrica()
    vistas = {dihedral(x, k, f).tobytes() for k, f in DIHEDRAL}
    assert len(DIHEDRAL) == 8 and len(vistas) == 8


def test_inversa_desfaz_e_numpy_igual_torch():
    x = _assimetrica()
    t = torch.from_numpy(x)
    for k, f in DIHEDRAL:
        y = dihedral_torch(t, k, f)
        assert np.array_equal(y.numpy(), dihedral(x, k, f))
        assert torch.equal(inverse_dihedral(y, k, f), t)


def test_copas_continuam_sobre_as_arvores():
    # Banda 0 marca as copas (valor 10 x etiqueta). Depois de qualquer
    # transformação sorteada, a marca tem de continuar sobre a mesma copa.
    copas = np.zeros((32, 32), np.int32)
    copas[2:8, 3:12] = 1
    copas[20:30, 15:18] = 2
    bandas = np.zeros((5, 32, 32), np.uint8)
    bandas[0] = copas * 10
    aug = RandomDihedral()
    torch.manual_seed(0)
    vistas = set()
    for _ in range(80):
        b, c = aug(bandas, copas)
        assert np.array_equal(b[0], c * 10)
        assert sorted(np.unique(c, return_counts=True)[1][1:]) == [30, 54]  # áreas
        vistas.add(c.tobytes())
    assert len(vistas) == 8                       # sorteou todas as 8


def test_lote_com_numeros_diferentes_de_copas():
    def item(n):
        return torch.zeros(6, 8, 8), {"boxes": torch.zeros(n, 4)}

    class Falso(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return item(i)                        # 0, 1 e 2 copas

    imgs, alvos = collate([item(0), item(3)])
    assert len(imgs) == 2 and [a["boxes"].shape[0] for a in alvos] == [0, 3]

    imgs, alvos = next(iter(make_loader(Falso(), batch_size=3, shuffle=False)))
    assert [a["boxes"].shape[0] for a in alvos] == [0, 1, 2]
