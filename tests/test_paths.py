"""Mensagens de arquivo faltando (rsskd/paths.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.paths import MissingFileError, checkpoints_near, require  # noqa: E402


def test_require_existe_e_falta(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}")
    assert require(f, "algo") == f
    with pytest.raises(FileNotFoundError, match="Rode antes: x.py"):
        require(tmp_path / "b.json", "algo", "Rode antes: x.py", ["a"])


def test_normalizacao_faltando_diz_o_script(tmp_path):
    with pytest.raises(MissingFileError, match="compute_input_stats"):
        InputSpec.load(tmp_path / "entrada.json")


def test_modelo_faltando_lista_os_que_existem(tmp_path):
    from rsskd.train import load_checkpoint

    run = tmp_path / "runs" / "curto_s0"
    (run / "epocas").mkdir(parents=True)
    (run / "ultimo.pt").write_bytes(b"")
    (run / "epocas" / "epoca_01.pt").write_bytes(b"")
    assert checkpoints_near(run / "melhor.pt") == ["ultimo", "epocas/epoca_01"]
    with pytest.raises(MissingFileError, match="ultimo, epocas/epoca_01"):
        load_checkpoint(run / "melhor.pt")
    # Treino com nome errado: sugere os treinos que existem.
    with pytest.raises(MissingFileError, match="curto_s0"):
        load_checkpoint(tmp_path / "runs" / "curto_s9" / "melhor.pt")
