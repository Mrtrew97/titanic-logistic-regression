# Modelo de Regressão Logística - Titanic

Este repositório contém a resolução do caso prático **"Modelo de Classificação com Regressão Logística"** no âmbito do módulo **Python para Análise de Dados** da **Master.D**.

O objetivo do projeto é construir um pipeline completo em Python para prever a probabilidade de sobrevivência dos passageiros do Titanic com base no dataset `titanic_v2.csv`.

---

## Visão Geral do Projeto

O projeto aborda todas as etapas fundamentais de ciência de dados e modelação estatística:

- **Carregamento e Exploração**: Leitura do CSV, verificação de caminhos com `pathlib`, estatísticas descritivas com `describe()`, interpretação dos padrões e exploração gráfica independente em Seaborn.
- **Tratamento e Validação de Nulos**: Imputação por mediana (`age`) e moda (`embarked`), acompanhada por verificação intermédia e final de zero nulos na consola.
- **Codificação Categórica Padronizada**: Aplicação consistente de `pd.get_dummies()` com `drop_first=True` em todas as variáveis categóricas (`sex` e `embarked`).
- **Divisão Estratificada de Amostras**: Separação dos dados utilizando `train_test_split()` do `scikit-learn` com `stratify=y` (80% treino / 20% teste) para preservar a proporção real de sobreviventes.
- **Modelação Estatística**: Treino do modelo de regressão logística binária (`sm.Logit`) com adição de constante exógena.
- **Avaliação no Conjunto de Teste**: Cálculo de probabilidades, limiar de decisão ($p \ge 0.5$), matriz de confusão e taxa global de acerto.
- **Geração de Relatório PDF**: Exportação automática de um relatório visual e interpretativo com gráficos criados em Seaborn.

### Seleção e Justificação de Variáveis

Para garantir a integridade e eficácia do modelo de regressão logística, realizou-se uma seleção criteriosa das variáveis do conjunto de dados `titanic_v2.csv`:

- **Variável Alvo (Target)**:
  - `survived`: Indicador binário de sobrevivência (0 = Não Sobreviveu, 1 = Sobreviveu).

- **Variáveis Mantidas (Preditores)**:
  - `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare` e `embarked`: Mantidas por representarem características demográficas, socioeconómicas e logísticas relevantes para a previsão da sobrevivência.

- **Variáveis Excluídas e Justificação**:
  - `cabin`: Excluída devido à elevada quantidade de valores nulos (~77,1% / 687 registos em falta), cuja imputação poderia introduzir ruído e distorção no modelo.
  - `passenger`, `name` e `ticket`: Excluídos por funcionarem essencialmente como identificadores individuais, não apresentando poder preditivo generalizável adequado ao objetivo do modelo.

---

## Estrutura do Repositório

- **`modelo_titanic.py`**: Script Python principal contendo todo o código executável.
- **`titanic_v2.csv`**: Ficheiro de dados com os registos históricos dos passageiros.
- **`relatorio_titanic.pdf`**: Documento PDF gerado automaticamente com os resultados, explicações pedagógicas e gráficos.
- **`README.md`**: Documentação principal do projeto.

---

## Requisitos e Instalação

### Dependências Necessárias

- Python 3.10 ou superior
- `numpy`
- `pandas`
- `statsmodels`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Comando de Instalação

```bash
python -m pip install -r requirements.txt
```

---

## Como Executar

1. Certifique-se de que o ficheiro `titanic_v2.csv` se encontra na mesma pasta do script.
2. Abra a linha de comandos ou terminal na pasta do projeto.
3. Execute o script principal:

```bash
python modelo_titanic.py
```

---

## Resultados e Avaliação

### Resultados do Modelo

- **Observações no Treino**: 712 passageiros
- **Observações no Teste**: 179 passageiros
- **Pseudo R²**: 0.3447 (indicando uma melhoria significativa de ajustamento face ao modelo nulo)
- **Significância Global (LLR p-value)**: $8.32 \times 10^{-66}$ ($p < 0.001$)

### Matriz de Confusão

A avaliação do modelo no conjunto de teste (179 novos passageiros) produziu os seguintes resultados:

- **Verdadeiros Negativos (Mortes Previstas Corretamente)**: 98
- **Verdadeiros Positivos (Sobrevivências Previstas Corretamente)**: 46
- **Falsos Positivos (Previu Sobrevivência, mas Faleceu)**: 12
- **Falsos Negativos (Previu Falecimento, mas Sobreviveu)**: 23

### Taxa Global de Acerto

- **Acurácia Geral**: **80.45%** nas previsões sobre dados de teste não vistos durante o treino.

### Análise Estatística

- **Variáveis Estatisticamente Significativas ($p < 0.05$)**:
  - `sex_male` ($p < 0.001$, $coef = -2.6908$): Ser do sexo masculino reduz significativamente a probabilidade de sobrevivência em comparação com o sexo feminino.
  - `pclass` ($p < 0.001$, $coef = -1.1419$): O aumento do valor de `pclass` está associado a uma menor probabilidade de sobrevivência, sendo a 3.ª classe a categoria de menor estatuto socioeconómico.
  - `age` ($p < 0.001$, $coef = -0.0396$): A idade avançada tem uma associação ligeiramente negativa com a probabilidade de sobrevivência.
  - `sibsp` ($p = 0.037$, $coef = -0.2531$): O número de irmãos/cônjuges a bordo apresenta associação negativa com a sobrevivência.

- **Variáveis Sem Significância Estatística ($p \ge 0.05$)**:
  - `parch`, `fare`, `embarked_Q` e `embarked_S` não demonstraram evidência estatística suficiente ao nível de 5% neste conjunto de dados (todos com $p \ge 0.05$).