"""Mensagens claras quando falta um arquivo ou uma pasta.

Sem isto, um arquivo que falta aparece como um erro genérico do Python
(FileNotFoundError do json, do torch ou do rasterio), sem dizer de onde o
arquivo deveria ter vindo. Aqui, cada erro diz o que falta e qual comando o
gera, e avisa quando o terminal não está na pasta do repositório (a causa
mais comum: os caminhos padrão, como data/ e configs/, são relativos a ela).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


HINT_FOLDER = ("Gere as previsões do aluno com scripts/export_predictions.py (vão para "
               "previsoes/<treino>); as copas do TEBAKA ficam em data/pseudo/agml "
               "(scripts/generate_pseudolabels.py).")


class MissingFileError(FileNotFoundError):
    """FileNotFoundError com a explicação de como gerar o arquivo."""


def _cwd_hint(path: Path) -> str:
    if path.is_absolute() or Path.cwd().resolve() == REPO:
        return ""
    return (f"\n  Atenção: o terminal está em {Path.cwd()}, e os caminhos são "
            f"relativos a ele. Rode os scripts da pasta do repositório ({REPO}).")


def require(path, what: str, hint: str = "", alternatives=()) -> Path:
    """Devolve o caminho se ele existe; senão, um erro que explica o que fazer.

    what:         o que é o arquivo, em poucas palavras ("normalização da entrada").
    hint:         como gerá-lo ("Rode antes: python scripts/...").
    alternatives: caminhos parecidos que existem, para sugerir no erro.
    """
    p = Path(path)
    if p.exists():
        return p
    msg = f"Não achei {what}: {p}"
    if hint:
        msg += f"\n  {hint}"
    alternatives = [str(a) for a in alternatives]
    if alternatives:
        msg += "\n  Existem: " + ", ".join(alternatives)
    raise MissingFileError(msg + _cwd_hint(p))


def checkpoints_near(path) -> list[str]:
    """Os .pt que existem na pasta do treino (e em epocas/), para sugerir."""
    folder = Path(path).parent
    if folder.name == "epocas":
        folder = folder.parent
    if not folder.exists():
        return []
    found = sorted(folder.glob("*.pt")) + sorted(folder.glob("epocas/*.pt"))
    return [str(f.relative_to(folder).with_suffix("")).replace("\\", "/") for f in found]
