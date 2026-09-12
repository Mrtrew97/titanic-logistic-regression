# Modelo de Regressão Logística - Titanic

Modelo de classificação desenvolvido em Python para prever a probabilidade de sobrevivência dos passageiros do Titanic, utilizando tratamento de dados, análise estatística e regressão logística.

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido no âmbito de um caso prático de **Python para Análise de Dados da Master.D** e demonstra a construção de um processo completo de análise e modelação de dados.

O programa utiliza o dataset **titanic_v2.csv** e permite:


- Carregar e explorar os dados dos passageiros.
- Identificar e tratar valores nulos.
- Preparar e transformar variáveis categóricas.
- Dividir os dados em conjuntos de treino e teste.
- Treinar um modelo de regressão logística binária.
- Prever a probabilidade de sobrevivência dos passageiros.
- Avaliar o desempenho do modelo através de uma matriz de confusão e da taxa de acerto.
- Analisar a significância estatística das variáveis utilizadas no modelo.
- Gerar um relatório PDF com os resultados e gráficos da análise.

---

### 📊 Seleção e Justificação de Variáveis

Para a construção do modelo de regressão logística, foram selecionadas as variáveis consideradas relevantes para a previsão da sobrevivência dos passageiros.

- **Variável Alvo (Target)**:
  - `survived`: Indicador binário de sobrevivência (0 = Não Sobreviveu, 1 = Sobreviveu).

- **Variáveis Preditoras**:
  - `pclass`: Classe do passageiro.
  - `sex`: Sexo do passageiro.
  - `age`: Idade.
  - `sibsp`: Número de irmãos ou cônjuges a bordo.
  - `parch`: Número de pais ou filhos a bordo.
  - `fare`: Tarifa do bilhete.
  - `embarked`: Porto de embarque.

- **Variáveis excluídas:**:
  - `cabin`: Excluída devido à elevada quantidade de valores nulos, correspondentes a aproximadamente 77,1% dos registos (687 valores em falta).
  - `passenger`, `name` e `ticket`: Excluídas por funcionarem essencialmente como identificadores individuais, não sendo consideradas variáveis com poder preditivo generalizável adequado ao objetivo do modelo.

---

## Estrutura do Repositório

- **.gitattributes:** Ficheiro de configuração utilizado pelo Git para definir atributos específicos dos ficheiros do repositório.
- **modelo_titanic.py: Script Python principal contendo todo o código executável do projeto.
- **titanic_v2.csv:** Dataset com os registos históricos dos passageiros do Titanic.
- **relatorio_titanic.pdf:** Relatório PDF gerado automaticamente com os resultados da análise, gráficos e interpretação do modelo.
- **requirements.txt:** Ficheiro com as bibliotecas Python necessárias para executar o projeto.
- **README.md:** Documentação principal do projeto.

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

### Instalação

Com as dependências definidas no ficheiro requirements.txt, execute:
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 Execução

1. Certifique-se de que o ficheiro `titanic_v2.csv` se encontra na mesma pasta do script.
2. Abra a linha de comandos ou terminal na pasta do projeto.
3. Execute o script principal:

```bash
python modelo_titanic.py
```

---

## 📊 Resultados e Avaliação

### Resultados do Modelo

- **Dados de treino**: 712 passageiros
- **Dados de teste**: 179 passageiros
- **Pseudo R²**: 0.3447 
- **LLR p-value**: 8.32 × 10⁻⁶⁶ (p < 0.001)

O resultado do teste de razão de verosimilhança (LLR) indica que o modelo apresenta significância estatística global.


### Matriz de Confusão

A avaliação do modelo no conjunto de teste produziu os seguintes resultados:

- **Verdadeiros Negativos (Mortes Previstas Corretamente)**: 98
- **Verdadeiros Positivos (Sobrevivências Previstas Corretamente)**: 46
- **Falsos Positivos (Previu Sobrevivência, mas Faleceu)**: 12
- **Falsos Negativos (Previu Falecimento, mas Sobreviveu)**: 23

### Taxa Global de Acerto

- **Acurácia**: **80.45%** 

A taxa de acerto foi calculada sobre os 179 passageiros do conjunto de teste, que não foram utilizados no treino do modelo.


### Análise Estatística

- **Variáveis Estatisticamente Significativas ($p < 0.05$)**:
  - `sex_male` ($p < 0.001$, $coef = -2.6908$): Ser do sexo masculino reduz significativamente a probabilidade de sobrevivência em comparação com o sexo feminino.
  - `pclass` ($p < 0.001$, $coef = -1.1419$): O aumento do valor de `pclass` está associado a uma menor probabilidade de sobrevivência, sendo a 3.ª classe a categoria de menor estatuto socioeconómico.
  - `age` ($p < 0.001$, $coef = -0.0396$): A idade avançada tem uma associação ligeiramente negativa com a probabilidade de sobrevivência.
  - `sibsp` ($p = 0.037$, $coef = -0.2531$): O número de irmãos/cônjuges a bordo apresenta associação negativa com a sobrevivência.

- **Variáveis Sem Significância Estatística ($p \ge 0.05$)**:
  - `parch`
  - `fare`
  - `embarked_Q`
  - `embarked_S`
Estas variáveis não apresentaram evidência estatística suficiente para serem consideradas significativas ao nível de 5% neste modelo.
