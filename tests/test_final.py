"""Avaliação final no teste (scripts/final_test.py)."""

import pytest

pytest.importorskip("torch")
from test_train import dados_falsos  # noqa: E402,F401  (fixture)


def test_avaliacao_final_com_sementes(dados_falsos, tmp_path):
    from final_test import main as final
    from run_seeds import main as rodar

    dados, pseudo, entrada = dados_falsos
    comum = ["--pasta", str(dados), "--copas", str(pseudo), "--entrada", str(entrada),
             "--lote", "1", "--sem-pretreino", "--dispositivo", "cpu",
             "--saida", str(tmp_path / "runs"), "--epocas", "1",
             "--max-passos-por-epoca", "1", "--paciencia", "0", "--peso-consistencia", "0"]
    rodar(["--nome", "m", "--sementes", "0", "1", "--"] + comum)
    args = [f"Aluno={tmp_path / 'runs' / 'm_s*'}", "--split", "val", "--pasta", str(dados),
            "--copas", str(pseudo), "--previsoes", str(tmp_path / "prev"),
            "--saida", str(tmp_path / "saidas"), "--dispositivo", "cpu", "--bootstrap", "200"]
    resumos, comparacoes = final(args)
    assert list(resumos) == ["TEBAKA", "Aluno"] and len(resumos["Aluno"]) == 2
    assert "Aluno" in comparacoes
    md = (tmp_path / "saidas" / "final_val.md").read_text(encoding="utf-8")
    assert "| TEBAKA | 1 |" in md and "| Aluno | 2 |" in md
    assert not (tmp_path / "saidas" / "registro_teste.csv").exists()   # só no teste
    # Rodar de novo não exporta de novo.
    antes = (tmp_path / "prev" / "m_s0" / "info.json").stat().st_mtime
    final(args)
    assert (tmp_path / "prev" / "m_s0" / "info.json").stat().st_mtime == antes
    with pytest.raises(FileNotFoundError, match="Nenhum treino"):
        final(["Outro=" + str(tmp_path / "runs" / "zzz*")] + args[1:])


def test_sementes_somadas_por_imagem():
    from final_test import pooled_rows

    from rsskd.evaluation import CrownMetrics
    import numpy as np

    ref = np.zeros((8, 8), np.int32)
    ref[:4, :4] = 1
    a, b = CrownMetrics(), CrownMetrics()
    a.add(ref, ref, index=0)
    b.add(np.zeros_like(ref), ref, index=0)
    (linha,) = pooled_rows([a, b])
    assert linha["tp"] == 1 and linha["fn"] == 1 and linha["n_ref"] == 2
