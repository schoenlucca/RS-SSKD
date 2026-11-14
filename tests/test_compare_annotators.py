"""Concordância entre rotuladores (scripts/compare_annotators.py)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")
from test_import_labels import _quadrado  # noqa: E402
from test_labeling import _voo  # noqa: E402


def test_concordancia_entre_rotuladores(tmp_path):
    from compare_annotators import main
    from export_for_labeling import main as exportar

    _voo(tmp_path / "proj", "T", 12)
    crops = exportar(["--voos", "T", "--origem", str(tmp_path / "proj"), "--n", "2",
                      "--espaco", "5", "--pasta", str(tmp_path / "c"),
                      "--saida", str(tmp_path / "rot")])
    imgs = [{"id": i + 1, "file_name": c.arquivo, "width": 512, "height": 512}
            for i, c in enumerate(crops)]
    # As duas pessoas marcam as mesmas duas copas na imagem 1; na imagem 2,
    # B marca uma copa que A não marcou. Só B rotulou uma terceira imagem.
    a = {"images": imgs, "annotations": [
        {"image_id": 1, "segmentation": _quadrado(100, 100, 100)},
        {"image_id": 1, "segmentation": _quadrado(300, 300, 100)},
        {"image_id": 2, "segmentation": _quadrado(0, 0, 200)}]}
    b = {"images": imgs + [{"id": 3, "file_name": "outra.png", "width": 512, "height": 512}],
         "annotations": [
        {"image_id": 1, "segmentation": _quadrado(104, 100, 100)},
        {"image_id": 1, "segmentation": _quadrado(300, 296, 100)},
        {"image_id": 2, "segmentation": _quadrado(0, 0, 200)},
        {"image_id": 2, "segmentation": _quadrado(300, 300, 100)}]}
    for nome, d in (("a", a), ("b", b)):
        (tmp_path / f"{nome}.json").write_text(json.dumps(d), encoding="utf-8")
    s, linhas = main([str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                      "--nomes", "Lucca", "Felipe",
                      "--manifesto", str(tmp_path / "rot" / "manifesto.json"),
                      "--saida", str(tmp_path / "s")])
    assert len(linhas) == 2                                     # só as imagens em comum
    assert s["f1"] == pytest.approx(2 * 3 / (3 + 4))            # 3 casadas, 3 x 4 copas
    assert 0.9 < s["iou_copa"] <= 1.0
    md = (tmp_path / "s" / "concordancia_rotulos.md").read_text(encoding="utf-8")
    assert "Lucca" in md and "Felipe" in md
