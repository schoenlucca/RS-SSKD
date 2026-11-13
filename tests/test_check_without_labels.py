"""Diagnóstico sem referência (scripts/check_without_labels.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from test_train import dados_falsos  # noqa: E402,F401  (fixture)


class CopiaCanal:
    """'Modelo' que prevê como copa a região onde o canal 0 é alto: a
    previsão gira junto com a imagem, então a estabilidade tem de ser 1."""

    def __call__(self, imgs):
        x = imgs[0]
        m = (x[0] > 0.5).float()[None]
        return [{"masks": m[None], "scores": torch.ones(1)}]


def test_rotacao_desfeita_volta_a_mesma_previsao():
    from check_without_labels import TRANSFORMS, f1_between, student_predictions

    img = torch.zeros(3, 40, 40)
    img[0, 5:15, 22:35] = 1.0                               # copa fora do centro
    base = student_predictions(CopiaCanal(), [img], "cpu")
    assert base[0].sum() == 10 * 13
    for t in TRANSFORMS:
        assert f1_between(student_predictions(CopiaCanal(), [img], "cpu", t), base) == 1.0


def test_script_com_duas_sementes(dados_falsos, tmp_path):
    from check_without_labels import main
    from run_seeds import main as rodar

    dados, pseudo, entrada = dados_falsos
    comum = ["--pasta", str(dados), "--copas", str(pseudo), "--entrada", str(entrada),
             "--lote", "1", "--sem-pretreino", "--dispositivo", "cpu",
             "--saida", str(tmp_path / "runs"), "--epocas", "1",
             "--max-passos-por-epoca", "1", "--paciencia", "0", "--peso-consistencia", "0"]
    rodar(["--nome", "c", "--sementes", "0", "1", "--"] + comum)
    linhas, pares = main([str(tmp_path / "runs" / "c_s*"), "--dispositivo", "cpu",
                          "--saida", str(tmp_path / "saidas"), "--n", "2"])
    assert [r["metodo"] for r in linhas] == ["TEBAKA", "c_s0", "c_s1"]
    assert all(0 <= r["estabilidade"] <= 1 for r in linhas[1:])
    assert list(pares) == ["c_s0 x c_s1"]
    md = (tmp_path / "saidas" / "sem_referencia_val.md").read_text(encoding="utf-8")
    assert "| TEBAKA |" in md and "Concordância entre sementes" in md
    with pytest.raises(FileNotFoundError):
        main([str(tmp_path / "runs" / "nada*")])

    # Com TTA: sem estabilidade (é 1 por construção), saídas com sufixo _tta.
    linhas, pares = main([str(tmp_path / "runs" / "c_s*"), "--dispositivo", "cpu",
                          "--saida", str(tmp_path / "saidas"), "--n", "2", "--tta"])
    assert "estabilidade" not in linhas[1] and list(pares) == ["c_s0 x c_s1"]
    assert (tmp_path / "saidas" / "sem_referencia_val_tta.md").exists()


def test_professor_e_quase_estavel_na_rotacao():
    from check_without_labels import TRANSFORMS, f1_between, teacher_rotated
    from test_teacher import NIR, RED, synthetic_image

    from rsskd.teacher import TebakaTeacher

    img, _ = synthetic_image()
    bands = np.moveaxis(img, -1, 0)
    prof = TebakaTeacher(nir_band=NIR, red_band=RED)
    base = [prof(bands)["instances"]]
    for t in TRANSFORMS:
        assert f1_between(teacher_rotated(prof, [bands], t), base) > 0.8
