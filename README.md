# RS-SSKD + TEBAKA: copas de árvore sem rótulos humanos

Segmentação de copas individuais de árvores (uma máscara por árvore) em
imagens multiespectrais de drone, **sem nenhuma anotação humana no treino**.
PIBIC 2026, Plano de Trabalho 1.2 (UnB, orientador Prof. Díbio Borges).

O professor TEBAKA (NDVI, limiar e watershed) gera as copas que treinam o
aluno, um Mask R-CNN com ResNet-101. O aluno aprende com os princípios do
RS-SSKD (consistência sob transformações e auto-destilação) e com a regra de
área de Harmon et al. (2022). A referência humana só entra na avaliação.
Enquanto as imagens do drone não chegam, tudo roda no conjunto AgML de
Peng et al. (2026).

O porquê de cada decisão, as equações e as referências estão no registro do
projeto; este README cobre só o "como rodar".

## Pipeline

1. **Entrada:** as 5 bandas do AgML (vermelho, verde, azul, NIR e
   provavelmente Red-Edge) mais o NDVI, 6 canais normalizados com a média e o
   desvio do treino.
2. **Professor TEBAKA** (`rsskd/teacher.py`): NDVI, máscara de vegetação e
   watershed geram as copas que substituem as anotações humanas.
3. **Aluno, etapa 1:** Mask R-CNN com ResNet-101 + FPN de 6 canais, treinado
   com as copas do professor e, opcionalmente, com a consistência do RS-SSKD
   nos mapas da RPN.
4. **Aluno, etapa 2:** um aluno novo aprende com as copas do TEBAKA, com as
   probabilidades do aluno da etapa 1 congelado (auto-destilação) e com a
   regra 1 de Harmon (copas grandes demais são penalizadas).
5. **Avaliação:** precisão, revocação e F1 por copa (IoU ≥ 0,5), IoU, erro de
   contagem e métricas de área, contra a referência.

## Instalação

No Windows, crie o ambiente **fora do OneDrive**: a sincronização corrompe a
instalação do PyTorch. Instale o PyTorch com CUDA **antes** do
`requirements.txt`, senão o pip instala a versão só de CPU.

```
python -m venv C:\venvs\rsskd
C:\venvs\rsskd\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A última linha deve terminar em `True`. No VS Code, escolha esse ambiente em
"Python: Select Interpreter" (`C:\venvs\rsskd\Scripts\python.exe`); um
terminal sem ele ativado dá `ModuleNotFoundError: torchvision`.

Rode todos os comandos **da pasta do repositório**: os caminhos padrão
(`data/`, `configs/`, `runs/`) são relativos a ela.

## Passo a passo

### 1. Dados e professor

```
python scripts/inspect_agml.py --baixar          # baixa o AgML (~330 MB) em data/agml_tree_crown/
python scripts/evaluate_teacher.py --split val   # professor na validação
python scripts/generate_pseudolabels.py          # copas do TEBAKA em data/pseudo/agml/
python scripts/summarize_pseudolabels.py         # professor x referência; mediana da área
python scripts/compute_input_stats.py            # normalização em configs/agml_entrada.json
```

Os parâmetros do professor já calibrados ficam em `configs/agml_teacher.json`.
Para calibrar de novo: `python scripts/evaluate_teacher.py --split val --grade`.

Para conferir a entrada antes de treinar:

```
python scripts/show_batch.py        # um lote como o treino o vê
python scripts/overfit_check.py     # o aluno decora 4 imagens? (alguns minutos)
```

### 2. Aluno, etapa 1

O treino curto (4 épocas) é o que usamos: depois das primeiras épocas, o aluno
passa a decorar os erros do professor.

```
python scripts/run_seeds.py --nome curto_l0 --sementes 0 1 2 -- --epocas 4 --peso-consistencia 0 --salvar-epocas
python scripts/run_seeds.py --nome curto_l1 --sementes 0 1 2 -- --epocas 4 --peso-consistencia 1 --salvar-epocas
```

`run_seeds.py` cria um treino por semente (`curto_l0_s0`, `_s1`, `_s2`), pula
os que já terminaram e retoma os interrompidos. Tudo depois de `--` vai para
`scripts/train_student.py`; `python scripts/train_student.py --help` lista as
opções.

### 3. Aluno, etapa 2

Com um aluno da etapa 1 como professor congelado:

```
python scripts/run_seeds.py --nome e2_kd --sementes 0 1 2 -- --professor runs/curto_l0_s0/melhor.pt --peso-consistencia 0 --paciencia 0
python scripts/run_seeds.py --nome e2_kd_regra --sementes 0 1 2 -- --professor runs/curto_l0_s0/melhor.pt --mu-max 0.3 --peso-consistencia 0 --paciencia 0
```

`--mu-max` liga a regra 1 de Harmon; `--peso-kd` e `--temperatura` controlam a
auto-destilação. A área de referência da regra é, por padrão, a mediana das
caixas das copas do TEBAKA no treino; `--estat-area p90` usa o percentil 90, para
a regra pesar só sobre as copas grandes demais (em geral, árvores vizinhas
fundidas pelo professor).

Na GPU, `--amp` treina em precisão mista (float16), mais rápido e com menos
memória.

### 4. Avaliar e comparar

```
python scripts/compare_runs.py                              # tabela de todos os treinos, com o TEBAKA na 1ª linha
python scripts/evaluate_student.py --run runs/curto_l0_s0   # um modelo salvo, com figura
python scripts/evaluate_student.py --run runs/curto_l0_s0 --modelo epocas/epoca_02
```

**O teste é para o fim.** Escolha parâmetros e modelos pela validação, sem
olhar a referência; rode o teste uma vez, só nas versões finais, com:

```
python scripts/final_test.py "Etapa 1=runs/curto_l0_s*" "Etapa 2=runs/e2_kd_s*"
```

Ele exporta as previsões de cada semente, mede tudo contra a referência e
grava em `saidas/teste_final.md` a média ± desvio entre sementes, a diferença
de F1 de cada método para o TEBAKA (bootstrap pareado nas imagens, com
intervalo de 95% e valor p) e o erro relativo de área nas copas que todos
acertaram. Cada execução no teste fica anotada em `saidas/registro_teste.csv`.
Com `--split val`, faz o mesmo na validação, sem anotar.

**Previsão com as 8 orientações (TTA).** Qualquer modelo salvo pode prever
cada imagem nas 4 rotações, com e sem espelho, e combinar as copas: o que
aparece em pelo menos 4 das 8 orientações fica, o resto sai
(`rsskd/tta.py`). Não precisa treinar de novo; é cerca de 8 vezes mais lento.

```
python scripts/evaluate_student.py --run runs/curto_l0_s0 --tta        # compara com a avaliação comum
python scripts/final_test.py "Etapa 2=runs/e2_kd_s*" --split val --tta
```

`export_predictions.py` e `check_without_labels.py` também aceitam `--tta`; as
saídas ganham o sufixo `_tta`. Usar ou não o TTA é uma escolha a fazer na
validação, antes do teste.

### 5. Comparar com outros métodos (o SAM do Felipe)

```
python scripts/export_predictions.py --run runs/e2_kd_s0 --divisoes val test
python scripts/evaluate_predictions.py data/pseudo/agml previsoes/e2_kd_s0 previsoes/felipe_sam --split test
python scripts/figure_predictions.py TEBAKA=data/pseudo/agml "Aluno=previsoes/e2_kd_s0" SAM=previsoes/felipe_sam --split test
```

Qualquer pasta no formato de `rsskd/predictions.py` (uma imagem de etiquetas
por recorte: 0 = fundo, 1..N = copas, em `.npz` ou PNG de 16 bits) é medida
com as mesmas métricas. A primeira pasta é a linha de base: a tabela traz o
intervalo de 95% do F1 de cada método e a diferença de cada um para ela, por
bootstrap pareado (`rsskd/stats.py`). A conferência pela soma das bandas avisa se as
imagens estiverem fora de ordem.

### 6. Ortomosaico do drone

Troque `orto.tif` pelo caminho do seu arquivo:

```
python scripts/predict_orthomosaic.py --run runs/e2_kd_s0 --imagem orto.tif
python scripts/predict_orthomosaic.py --run runs/e2_kd_s0 --bandas R.tif G.tif B.tif NIR.tif RE.tif
python scripts/predict_orthomosaic.py --professor --imagem orto.tif
```

As bandas devem seguir a ordem do AgML. A saída é um GeoTIFF de etiquetas
alinhado à imagem e um CSV com uma linha por copa (área em m² e coordenadas
do centro).

### 7. Fotos do Cerrado (DJI Mavic 3M)

As fotos do laboratório (`projeto_cerrado/<voo>/processed-*_MS_{G,R,RE,NIR}.TIF`)
viram o mesmo formato do AgML, e o resto do pipeline roda com outra `--pasta`
e outra configuração. No servidor do LISA:

```
python3 scripts/prepare_cerrado.py                       # data/cerrado: voos de chuva, divisão por voo
python3 scripts/tune_teacher.py --grade split_by=nir th=0.05,0.15 min_distance=20,30 sigma=4,6
python3 scripts/tune_teacher.py --grade ... --gravar N   # grava a linha escolhida
python3 scripts/generate_pseudolabels.py --pasta data/cerrado --config configs/cerrado_teacher.json --saida data/pseudo/cerrado --divisoes train val
python3 scripts/compute_input_stats.py --pasta data/cerrado --config configs/cerrado_teacher.json --saida configs/cerrado_entrada.json
python3 scripts/run_seeds.py --nome cerrado_l0 --sementes 0 1 -- --pasta data/cerrado --copas data/pseudo/cerrado --entrada configs/cerrado_entrada.json --epocas 4 --peso-consistencia 0 --amp
python3 scripts/check_without_labels.py "runs/cerrado_l0_s*"
```

- **Entrada:** só as 4 bandas do Mavic 3M (verde, vermelho, Red-Edge, NIR) mais
  o NDVI, 5 canais. A foto RGB vem de outra câmera e fica desalinhada de 0,3
  a 2,6 pixels, por isso o azul não entra.
- **Escala e recortes:** cada foto de 1000×1000 é reduzida pela metade e
  cortada em 4 recortes de 256, para as copas grandes caberem inteiras.
- **Professor na mata fechada:** quase não há solo entre as copas, então o
  limiar de NDVI é fixo e baixo (0,15), e as copas são separadas pelo brilho
  do NIR (`split_by=nir`), não pela distância ao fundo. Os parâmetros foram
  escolhidos a olho, em recortes de treino (`tune_teacher.py`), porque não há
  referência.
- **Sem referência:** até existirem os rótulos à mão, o
  `check_without_labels.py` mede o que dá: a concordância com o professor, a
  estabilidade da previsão sob rotação e espelhamento e a concordância entre
  as sementes. As máscaras de referência podem trazer um número por copa
  (PNG de 16 bits), e as copas encostadas continuam separadas na avaliação.

### 8. Referência à mão no Cerrado

Os rótulos à mão viram a divisão de teste do Cerrado. O protocolo (o que é
copa, copas que se tocam, ferramenta) está em `docs/protocolo_rotulacao.md`.

```
python3 scripts/export_for_labeling.py --voos <voo do teste> --n 50          # rotulagem/imagens + manifesto.json
# ... rotular no CVAT ou no Roboflow (polígonos) e exportar em COCO ...
python3 scripts/compare_annotators.py rotulagem/lucca.json rotulagem/felipe.json --nomes Lucca Felipe
python3 scripts/import_labels.py --coco rotulagem/rotulos.json --rotulador "Lucca e Felipe"
python3 scripts/generate_pseudolabels.py --pasta data/cerrado --config configs/cerrado_teacher.json --saida data/pseudo/cerrado --divisoes test
python3 scripts/final_test.py "Etapa 1=runs/cerrado_l0_s*" --pasta data/cerrado --copas data/pseudo/cerrado
```

- Os recortes de rotulação têm 512×512 pixels da foto original, para quem
  rotula ver o detalhe. Na importação, os polígonos são reduzidos para 256×256,
  e cada recorte rotulado cai exatamente sobre um recorte do modelo.
- Os recortes saem de voos fora do treino e da validação, com disparos
  espaçados (fotos seguidas se sobrepõem). Os dois scripts recusam voos do
  treino.
- A referência guarda um número por copa: copas encostadas continuam separadas.
- `compare_annotators.py` mede a concordância entre duas pessoas nas mesmas
  imagens, que é o teto realista do F1.

## O que sai em cada pasta

Nenhuma dessas pastas vai para o git.

| Pasta | Conteúdo |
| --- | --- |
| `data/agml_tree_crown/` | Conjunto AgML baixado |
| `data/pseudo/agml/` | Copas do TEBAKA por divisão e `resumo.json` |
| `data/cerrado/`, `data/pseudo/cerrado/` | Recortes do Cerrado (`manifesto.json`) e as copas do professor neles |
| `runs/<treino>/` | `config.json`, `historico.csv` (perdas e métricas por época), `melhor.pt` (pelo critério do treino), `ultimo.pt`, `epocas/epoca_NN.pt` (com `--salvar-epocas`), `avaliacao_*` |
| `previsoes/<método>/` | Copas exportadas e `info.json` |
| `rotulagem/` | Recortes para rotular (`imagens/`, fora do git), `manifesto.json` e os arquivos COCO (vale versionar estes dois) |
| `saidas/` | Tabelas (`comparacao.md`, `previsoes_*.csv`), figuras e checagens |

`melhor_referencia.pt` é o melhor modelo contra a referência e serve **só para
diagnóstico**: escolher por ele usaria rótulos humanos.

## Resultados até agora (validação do AgML, 59 imagens)

| Método | P | R | F1 | Erro de contagem | KL área |
| --- | --- | --- | --- | --- | --- |
| TEBAKA (linha de base) | 0,52 | 0,47 | 0,49 | 3,5 | 0,50 |
| Aluno etapa 1, 4 épocas (3 sementes) | 0,47 | 0,56 | 0,51 | 4,7 | 0,12 |
| Aluno etapa 1, 20 épocas | 0,48 | 0,50 | 0,49 | 3,1 | 0,11 |
| Aluno etapa 2, auto-destilação, 4 épocas (3 sementes) | 0,48 | 0,57 | 0,52 | 4,3 | 0,10 |
| Aluno etapa 2, auto-destilação, 20 épocas (1 semente) | 0,52 | 0,54 | 0,53 | 3,3 | 0,11 |

Com 4 épocas, o aluno supera o professor nas 3 sementes (F1 0,507 ± 0,003): acha
mais copas e reproduz muito melhor a distribuição de tamanhos. A consistência do
RS-SSKD não mudou o F1 e dobra o tempo por época (~170 s contra ~82 s numa
RTX 3050). Na etapa 2, a auto-destilação (com o aluno de 4 épocas como
professor) sobe o F1 para 0,521 ± 0,003, e com ela o treino longo deixa de
cair; a regra 1 com a mediana não mudou o F1. No teste, o TEBAKA tem F1 0,52.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `rsskd/teacher.py`, `rsskd/pseudolabels.py` | Professor TEBAKA e suas copas |
| `rsskd/data.py` | Leitura do AgML (e da referência com um número por copa), casamento de copas |
| `rsskd/inputs.py`, `rsskd/dataset.py`, `rsskd/augment.py` | Entrada de 6 canais, conjunto e transformações D4 |
| `rsskd/detector.py` | O aluno (Mask R-CNN, ResNet-101 + FPN) |
| `rsskd/train.py` | Passo de treino das etapas 1 e 2 |
| `rsskd/consistency.py`, `rsskd/distill.py`, `rsskd/rules.py` | Consistência, auto-destilação e regra 1 |
| `rsskd/evaluation.py` | Métricas por copa e de área (incluindo o erro relativo nas copas em comum) |
| `rsskd/stats.py` | Intervalos de confiança e comparação pareada por bootstrap |
| `rsskd/predictions.py`, `rsskd/tiling.py` | Previsões em arquivo e ortomosaicos |
| `rsskd/tta.py` | Previsão com as 8 orientações combinadas |
| `rsskd/cerrado.py` | Leitura das fotos do Mavic 3M no formato do AgML |
| `rsskd/labeling.py`, `rsskd/coco.py` | Recortes para rotular e importação dos polígonos (COCO) |
| `rsskd/paths.py` | Mensagens claras quando falta um arquivo |
| `rsskd/model.py`, `rsskd/losses.py` | RS-SSKD original de classificação, mantido como referência |
| `scripts/` | Um script por passo acima |
| `docs/protocolo_rotulacao.md` | Protocolo de rotulação das copas |
| `configs/` | Parâmetros do professor e normalização da entrada (AgML e Cerrado) |
| `tests/` | Testes (`pytest`) |

## Testes

```
pytest
```

Os testes usam imagens sintéticas e não precisam do AgML nem de GPU.

## Referências principais

- Zhang, P. et al. RS-SSKD: Self-Supervision Equipped with Knowledge
  Distillation for Few-Shot Remote Sensing Scene Classification. Sensors,
  21(5), 1566, 2021.
- Harmon, I. et al. Injecting Domain Knowledge Into Deep Neural Networks for
  Tree Crown Delineation. IEEE TGRS, 60, 4415419, 2022.
- Epifani, L.; D'Avino, V.; Caruso, A. TEBAKA: Territorial Basic Knowledge
  Acquisition. An Agritech Project for Italy: Results on Self-Supervised
  Semantic Segmentation. IEEE ISCC, 2023.
- Peng et al. Conjunto multiespectral de copas (Project-AgML/tree_crown_segmentation).
  Ecological Informatics, 2026.
- He, K. et al. Mask R-CNN. ICCV, 2017.
