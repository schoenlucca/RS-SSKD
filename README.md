# RS-SSKD para segmentação de copas no Cerrado

Segmentação de copas individuais de árvores (uma máscara por árvore) em
imagens multiespectrais de drone do Cerrado, **sem anotação humana no treino**.
PIBIC 2026, Plano de Trabalho 1.2 (UnB, orientador Prof. Díbio Borges).

## Pipeline

1. **Dados:** imagens de 4 bandas (inclui o NIR), com o NDVI como canal extra.
   As máscaras de referência servem só para avaliar.
2. **Professor TEBAKA** (`rsskd/teacher.py`): NDVI, limiar e morfologia geram a
   máscara de vegetação, que é separada em copas individuais. Essas copas
   substituem as anotações humanas.
3. **Aluno, etapa 1:** Mask R-CNN com backbone ResNet-101 (5 canais), treinado
   com as copas do professor e com a consistência sob transformações do RS-SSKD.
4. **Aluno, etapa 2:** auto-destilação do RS-SSKD (professor congelado) e
   professor de regras de conhecimento de domínio (Harmon et al., 2022).
5. **Avaliação:** precisão, revocação e F1 por copa (IoU >= 0,5), erro de
   contagem e métricas de área, contra as máscaras de referência.

## Estrutura

| Caminho | Conteúdo | Situação |
| --- | --- | --- |
| `rsskd/teacher.py` | Professor TEBAKA | Pronto |
| `rsskd/data.py` | Leitura do conjunto de teste (AgML) e métricas por copa | Pronto |
| `rsskd/model.py` | ResNet com 5 canais e CAM | Será adaptado para Mask R-CNN |
| `rsskd/losses.py` | Consistência e auto-destilação | Será adaptado para máscaras |
| `scripts/` | Scripts de execução e visualização | Em construção |
| `tests/` | Testes | |

O código da fase de classificação few-shot (protótipos e episódios) saiu do
projeto e está no histórico do git: `git show b06a16f:rsskd/fewshot.py`.

## Resultados do professor TEBAKA (conjunto AgML)

Conjunto de teste de Peng et al. (2026), plantação de *Camellia oleifera*,
5 bandas em 8 bits: 0, 1 e 2 = vermelho, verde e azul (iguais ao RGB pronto);
3 = NIR (pela razão copa/fundo e pela baixa correlação com o vermelho);
4 = provavelmente Red-Edge. O NDVI usa as bandas 3 e 0. Como as bandas não
estão em reflectância, o NDVI fica quase todo negativo.

Parâmetros calibrados na validação (busca em grade): limiar -0,42,
distância mínima 18 px, área mínima 300 px.

| Divisão | Precisão | Revocação | F1 | IoU por copa | IoU por pixel | Erro de contagem |
| --- | --- | --- | --- | --- | --- | --- |
| Validação (59 imagens) | 0,52 | 0,47 | 0,49 | 0,65 | 0,59 | 3,5 |
| Teste (60 imagens) | 0,53 | 0,50 | 0,52 | 0,65 | 0,56 | 3,6 |

Reproduzir: `python scripts/evaluate_teacher.py --split test`.

## Instalação e testes

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest
```

Para treinar com GPU, instale o PyTorch com CUDA seguindo https://pytorch.org.

## Referências principais

- Zhang, P. et al. RS-SSKD: Self-Supervision Equipped with Knowledge
  Distillation for Few-Shot Remote Sensing Scene Classification. Sensors,
  21(5), 1566, 2021.
- Harmon, I. et al. Injecting Domain Knowledge Into Deep Neural Networks for
  Tree Crown Delineation. IEEE TGRS, 60, 4415419, 2022.
- Epifani, L.; D'Avino, V.; Caruso, A. TEBAKA: Territorial Basic Knowledge
  Acquisition. An Agritech Project for Italy: Results on Self-Supervised
  Semantic Segmentation. IEEE ISCC, 2023.
