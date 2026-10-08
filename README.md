# Treinamento observável — MLP com indicadores de P&D da UNESCO

Atividade prática avaliativa — Sistemas de Informação / Unitins
Autora: Gabriela Dorneles Brasil Santos Andrade

## Problema

**A participação feminina na pesquisa acadêmica e governamental de um país prevê a participação feminina no setor empresarial?**

- **Alvo:** `FRESP.SP.THC.BUSENTSP` — pesquisadoras como % do total de pesquisadores (contagem de pessoas, HC) no setor empresarial, em 3 classes: baixa (< 25%), intermediária (25–40%) e alta (≥ 40%).
- **Entradas:** a mesma proporção nos setores de ensino superior (`FRESP.SP.THC.HIEDUSP`), governo (`FRESP.SP.THC.GOVSP`) e instituições privadas sem fins lucrativos (`FRESP.SP.THC.PRNPSP`).
- **Dataset final:** 106 países (36 baixa, 52 intermediária, 18 alta).

Indicadores do arquivo **não usados**, por vazamento de informação:
- `FRESP.THC` e `FRESP.TFTE`: total do país, que já contém o setor empresarial;
- `FRESP.SP.TFTE.BUSENTSP`: o próprio alvo medido em tempo integral (FTE);
- demais versões FTE: redundantes com as versões HC.

Como as classes são desbalanceadas, a comparação entre runs usa o **F1 macro** além da acurácia.

Fonte: UNESCO DataHub, *R&D – Other Policy Relevant Indicators (OPRI)*, identificador `uis005`, formato CSV.

O treinamento é configurável (arquivos YAML), rastreável (MLflow Tracking com parâmetros, métricas por época e artefatos) e observável (MLflow Tracing com os spans `preparar`, `treinar` e `validar`).

## Estrutura

```
config/base.yaml            parâmetros comuns: arquitetura, split, semente, épocas
config/run_a.yaml           A — referência   (lr 0.01,  L2 0)
config/run_b.yaml           B — taxa menor   (lr 0.001, L2 0)
config/run_c.yaml           C — com L2       (lr 0.01,  L2 1e-5)
config/run_incidente.yaml   D — taxa alta (lr 0.5); não foi usada nesta entrega
preparar_dados.py           CSV da UNESCO -> tabela país × indicador + classe
treinar.py                  treino PyTorch + MLflow (uma run por configuração)
rodar_experimentos.py       um comando, três configurações
avaliar_teste.py            avaliação ÚNICA no teste reservado da run escolhida
verificar_traces.py         lista os traces gravados e a run de cada um
```

## Como reproduzir

### 1. Ambiente

Python 3.10 ou superior. Versões usadas nesta entrega: Python 3.13 e MLflow 3.16.1, no Windows (PowerShell).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, rode antes `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 2. Dados

Baixe o CSV em data.unesco.org (dataset `uis005`, Export → CSV) e salve como `data/raw/opri.csv`.

Liste os indicadores e quantos países têm dado:

```powershell
python preparar_dados.py --entrada data/raw/opri.csv --listar
```

Gere o dataset (uma linha só):

```powershell
python preparar_dados.py --entrada data/raw/opri.csv --alvo FRESP.SP.THC.BUSENTSP --entradas FRESP.SP.THC.HIEDUSP FRESP.SP.THC.GOVSP FRESP.SP.THC.PRNPSP --limites 25 40 --nomes_classes baixa intermediaria alta --justificativa "totais e versão FTE do setor empresarial excluídos por conterem o alvo"
```

Decisões tomadas nesta etapa (registradas em `data/processed/resumo_dados.json`, que também vai como artefato de cada run):

- usa o valor **mais recente** de cada país a partir de 2010;
- entradas escolhidas manualmente, com a justificativa registrada no resumo;
- países sem o valor do alvo são descartados; faltantes nas entradas ficam para a imputação;
- **não** imputa nem padroniza aqui: isso é feito em `treinar.py`, ajustado apenas no treino.

### 3. Rodar as três runs (um comando)

```powershell
python rodar_experimentos.py
```

Arquitetura (MLP 16-16), split (60/20/20 estratificado), semente (42) e épocas (300) são iguais nas três runs. Só mudam taxa de aprendizado e L2.

### 4. Run do diagnóstico (parada antecipada)

```powershell
python treinar.py --config config/run_a.yaml --epocas 9 --nome_run E-parada-antecipada
```

Mesma configuração da run A, com 9 épocas (a melhor `val_loss` de A foi na época 8, contando a partir de 0).

### 5. Ver e comparar no MLflow

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --workers 1
```

O `--workers 1` evita o erro `WinError 10022` no Windows. Abra http://127.0.0.1:5000, entre no experimento `mlp-unesco-rd-opri`, escolha **Model training** → **Runs** e use a visão de gráficos para comparar as curvas.

Para conferir os traces pelo terminal:

```powershell
python verificar_traces.py
```

### 6. Avaliar o modelo escolhido no teste (uma vez)

```powershell
python avaliar_teste.py --run_id <RUN_ID_ESCOLHIDA>
```

O script grava `test_accuracy`, `test_f1_macro`, `test_loss` e a matriz de confusão do teste na própria run e marca a tag `avaliado_no_teste=sim`. Uma segunda execução é recusada.

## O que cada run registra

| Tipo | Conteúdo |
|---|---|
| Parâmetros | taxa de aprendizado, L2, épocas, arquitetura, semente, frações do split, estratificação, dataset |
| Tags | commit do git, versões de Python, PyTorch e MLflow |
| Métricas por época | `train_loss`, `val_loss`, `val_accuracy` |
| Métricas finais | `val_f1_macro_final`, `val_accuracy_final`, `melhor_epoca_val_loss`, `gap_perda_val_menos_treino` |
| Artefatos | `modelo.pt`, `preprocessamento.joblib`, `resumo_preprocessamento.json`, `curvas.png`, `matriz_confusao_validacao.png`, `config_efetiva.yaml`, `resumo_dados.json` |
| Trace | `pipeline_treinamento` → `preparar`, `treinar`, `validar` (duração, entradas, saídas e erros) |

## Resultados

Validação com 21 países, mesmo split em todas as runs.

| Run | lr | L2 | Épocas | Melhor época (val_loss) | val_loss final | val_accuracy | val_F1 macro | Observação das curvas |
|---|---|---|---|---|---|---|---|---|
| A | 0.01 | 0 | 300 | 8 | 7,5600 | 0,429 | 0,397 | train_loss perto de zero; val_loss sobe sem parar depois do início (overfitting) |
| B | 0.001 | 0 | 300 | 83 | 1,1696 | 0,476 | 0,361 | train_loss desce devagar; val_loss quase estável perto de 1,0 |
| C | 0.01 | 1e-5 | 300 | 8 | 10,9169 | 0,429 | 0,344 | igual a A: o L2 de 1e-5 foi fraco demais para frear o overfitting |
| E | 0.01 | 0 | 9 | — | 1,0165 | 0,524 | 0,367 | run A parada na melhor época |

As execuções foram repetidas do zero e reproduziram exatamente os mesmos valores de validação.

**Modelo escolhido: E-parada-antecipada**, por ter a menor `val_loss`. A acurácia e o F1 não foram usados para desempatar, porque as diferenças entre as runs equivalem a 1 ou 2 países.

**Teste reservado (22 países), avaliado uma única vez na run E:** acurácia 0,500, F1 macro 0,364, perda 1,0402.

| Classe | precision | recall | f1-score | support |
|---|---|---|---|---|
| alta | 0,00 | 0,00 | 0,00 | 4 |
| baixa | 0,50 | 0,57 | 0,53 | 7 |
| intermediaria | 0,50 | 0,64 | 0,56 | 11 |

## Diagnóstico de incidentes

**1. Overfitting nas runs A e C**
- Sintoma: `val_loss` subindo continuamente enquanto `train_loss` vai a quase zero.
- Hipótese: treino longo demais para a taxa de 0.01 (melhor época = 8 de 300).
- Nova run: E, igual a A com 9 épocas. A `val_loss` caiu de 7,56 para 1,02, confirmando a hipótese.

**2. Traces sobrescritos**
- Sintoma: o MLflow mostrava 1 trace para 4 runs, com os spans duplicados.
- Hipótese: o ID do trace era sempre o mesmo, `bdd640fb06671ad11c80317fa3b1799d`, que é o primeiro número gerado pelo módulo `random` do Python depois de `random.seed(42)`. O MLflow usa esse gerador para criar os IDs, e o `treinar.py` fixava essa semente no início de cada run.
- Correção: o `random` do Python deixou de ser semeado (o treino não depende dele). Depois de rodar tudo de novo, `verificar_traces.py` mostrou 4 traces, um por run.

## Limitações

- Com 21 países na validação e 22 no teste, uma diferença de 1 país muda a acurácia em cerca de 5 pontos percentuais. Diferenças pequenas entre runs são inconclusivas.
- A acurácia no teste (0,500) é igual à de um classificador que sempre responde a classe mais frequente (11 de 22 países são "intermediaria"). O modelo não acertou nenhum país da classe "alta". As 3 entradas têm pouco poder preditivo para o alvo.
- O número de épocas da run E foi escolhido olhando a validação, por isso o resultado de validação dela pode ser otimista. O teste reservado serve para essa verificação: a acurácia, o F1 e a perda no teste ficaram próximos dos da validação.

.venv/
__pycache__/
mlruns/
*.db
