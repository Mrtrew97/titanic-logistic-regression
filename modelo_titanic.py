# Importação das bibliotecas necessárias para carregamento de dados, análise estatística, modelação e visualização.
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Script para carregar e explorar os dados do Titanic, preparar os dados, treinar o modelo e gerar o relatório PDF.

BASE_DIR = Path(__file__).resolve().parent

# Define as variáveis utilizadas na análise e no modelo, excluindo identificadores e 'cabin' devido aos valores nulos.
COLUNAS_RELEVANTES = [
    'survived',
    'pclass',
    'sex',
    'age',
    'sibsp',
    'parch',
    'fare',
    'embarked',
]


# Função responsável por carregar o dataset Titanic a partir do ficheiro CSV.
def carregar_dados(caminho_ficheiro):
  caminho = Path(caminho_ficheiro)
  if not caminho.exists():
    raise FileNotFoundError(f"O ficheiro '{caminho}' não foi encontrado.")
  return pd.read_csv(caminho)


def explorar_dados(df_titanic):
  # Função para inspecionar a estrutura do dataset, tipos de dados, estatísticas descritivas e valores nulos.
  print('\n=== Exploração Inicial do Dataset ===')
  print(f'Forma do dataset: {df_titanic.shape}')
  print('\nPrimeiras linhas do dataset:')
  print(df_titanic.head())
  print('\nTipos de dados por coluna:')
  print(df_titanic.dtypes)
  print('\nEstatísticas descritivas das variáveis numéricas (describe):')
  print(df_titanic.describe())
  print('\nValores nulos por coluna (antes do tratamento):')
  print(df_titanic.isnull().sum())

  print('\n=== Interpretação dos Padrões Observados ===')
  print(
      '1. Disparidade por Género: As mulheres apresentaram uma taxa de'
      ' sobrevivência de ~74.2%, enquanto os homens tiveram apenas ~18.9%.'
  )
  print(
      '2. Impacto da Classe: A 1ª classe teve ~63.0% de sobrevivência,'
      ' comparada com ~47.3% na 2ª classe e apenas ~24.2% na 3ª classe.'
  )
  print(
      '3. Imputação de Nulos: A variável "age" contém 177 nulos (~19.8%) e'
      ' "embarked" contém 2 nulos, justificando a imputação.'
  )
  print(
      '4. Distribuição Etária e Tarifas: Idade média de ~29.7 anos e tarifas com'
      ' forte assimetria (valores entre 0 e 512.33).'
  )


def plotar_taxa_sobrevivencia_por_classe_sexo(df_titanic, ax):
  """Desenha o barplot de taxa de sobrevivência por classe e sexo no eixo
  fornecido. Função partilhada entre a exploração gráfica inicial e o
  relatório PDF, para evitar duplicar a lógica de plotagem e a preparação
  da coluna 'sex_label' em dois sítios diferentes."""
  df_temp = df_titanic.copy()
  df_temp['sex_label'] = df_temp['sex'].map(
      {'female': 'Feminino', 'male': 'Masculino'}
  )

  sns.barplot(
      data=df_temp,
      x='pclass',
      y='survived',
      hue='sex_label',
      palette='Set2',
      ax=ax,
  )
  ax.set_title('Taxa de Sobrevivência por Classe e Sexo')
  ax.set_xlabel('Classe do Bilhete (1ª, 2ª, 3ª Classe)')
  ax.set_ylabel('Proporção de Sobrevivência (0 a 1)')
  ax.set_ylim(0, 1)


# Função para explorar graficamente os dados e identificar padrões antes da modelação.
def explorar_dados_graficamente(df_titanic):
  sns.set_style('whitegrid')
  figura_exploratoria, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  sns.histplot(
      df_titanic['age'].dropna(),
      kde=True,
      ax=ax1,
      color='skyblue',
      bins=30,
  )
  ax1.set_title('1. Distribuição Etária dos Passageiros (Age)')
  ax1.set_xlabel('Idade (Anos)')
  ax1.set_ylabel('Frequência Absoluta')

  plotar_taxa_sobrevivencia_por_classe_sexo(df_titanic, ax2)
  ax2.set_title('2. ' + ax2.get_title())

  plt.tight_layout()
  plt.show()

# Função que compara os valores nulos antes e depois do tratamento.
def validar_tratamento_nulos(df_dados_antes, df_dados_depois):
  print('\n=== Validação do Tratamento de Valores Nulos ===')
  print('Valores nulos antes da imputação:')
  print(df_dados_antes.isnull().sum())
  print('\nValores nulos após a imputação (mediana em age, moda em embarked):')
  print(df_dados_depois.isnull().sum())
  print(
      '\nVerificação final do total de nulos:'
      f' {df_dados_depois.isnull().sum().sum()} (0 nulos confirmados)'
  )


# Função que seleciona e prepara os dados para o modelo.
def preparar_dados(df_titanic):
  df_dados_antes = df_titanic[COLUNAS_RELEVANTES].copy()
  df_dados_depois = df_dados_antes.copy()

  # Preenche os valores nulos de 'age' com a mediana para reduzir o impacto de valores extremos.
  df_dados_depois['age'] = df_dados_depois['age'].fillna(
      df_dados_depois['age'].median()
  )
  # Preenche os valores nulos de 'embarked' com a categoria mais frequente.
  df_dados_depois['embarked'] = df_dados_depois['embarked'].fillna(
      df_dados_depois['embarked'].mode()[0]
  )

  validar_tratamento_nulos(df_dados_antes, df_dados_depois)

  df_imputado = df_dados_depois.copy()

  # Converte as variáveis categóricas em colunas numéricas através de one-hot encoding.
  colunas_categoricas = ['sex', 'embarked']
  df_dados_depois = pd.get_dummies(
      df_dados_depois, columns=colunas_categoricas, drop_first=True, dtype=int
  )

  return df_dados_depois, df_imputado


# Função que separa os dados em conjuntos de treino e teste.
def dividir_dados(df_dados):
  y = df_dados['survived']
  X = df_dados.drop(columns=['survived'])

  # Divide 80% dos dados para treino e 20% para teste, mantendo a proporção da variável alvo.
  X_treino_raw, X_teste_raw, variavel_resposta_treino, variavel_resposta_teste = (
      train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  )

  # Adiciona uma constante ao modelo para representar o valor inicial.
  preditores_treino = sm.add_constant(X_treino_raw, prepend=False)
  preditores_teste = sm.add_constant(X_teste_raw, prepend=False)

  return (
      X_treino_raw,
      X_teste_raw,
      variavel_resposta_treino,
      variavel_resposta_teste,
      preditores_treino,
      preditores_teste,
  )


# Função que treina o modelo de regressão logística com os dados de treino.
def treinar_modelo(variavel_resposta_treino, preditores_treino):
  # Cria o modelo de regressão logística para estimar a probabilidade de sobrevivência.
  modelo_logit = sm.Logit(variavel_resposta_treino, preditores_treino)
  return modelo_logit.fit()


# Função que avalia o desempenho do modelo nas previsões do conjunto de teste.
def avaliar_modelo(resultado_modelo, variavel_resposta_teste, preditores_teste):
  probabilidades_teste = resultado_modelo.predict(preditores_teste)
  # Converte as probabilidades previstas em classes binárias utilizando 0.5 como limiar.
  previsoes_binarias = [
      1 if probabilidade >= 0.5 else 0
      for probabilidade in probabilidades_teste
  ]

  # Constrói a matriz de confusão para identificar acertos e erros das previsões.
  matriz_confusao = pd.crosstab(
      index=variavel_resposta_teste,
      columns=pd.Categorical(previsoes_binarias),
      rownames=['Observado'],
      colnames=['Previsto'],
  )

  # Calcula a percentagem de previsões corretas no conjunto de teste.
  taxa_acerto_global = np.mean(variavel_resposta_teste == previsoes_binarias)

  return (
      probabilidades_teste,
      previsoes_binarias,
      matriz_confusao,
      taxa_acerto_global,
  )


# Função que cria um relatório em PDF com os resultados e gráficos do modelo.
def gerar_relatorio_pdf(
    dados_treino,
    dados_teste,
    matriz_confusao,
    taxa_acerto_global,
    resultado_modelo,
    df_imputado,
):
  sns.set_style('whitegrid')

  # Cria uma figura em branco (sem eixos) para servir de "página" de texto no PDF.
  figura_texto_resumo, ax_texto = plt.subplots(figsize=(8.5, 11))
  ax_texto.axis('off')

  # Extrai as métricas globais de ajuste do modelo.
  pseudo_r2 = resultado_modelo.prsquared
  llr_pvalue = resultado_modelo.llr_pvalue

  # Extrai os coeficientes das principais variáveis para a interpretação.
  coef_sex = resultado_modelo.params['sex_male']
  coef_pclass = resultado_modelo.params['pclass']
  coef_age = resultado_modelo.params['age']
  coef_sibsp = resultado_modelo.params['sibsp']

  # Extrai os p-values correspondentes, para avaliar a significância estatística.
  pvalue_sex = resultado_modelo.pvalues['sex_male']
  pvalue_pclass = resultado_modelo.pvalues['pclass']
  pvalue_age = resultado_modelo.pvalues['age']
  pvalue_sibsp = resultado_modelo.pvalues['sibsp']

  # Separa as variáveis em significativas e não significativas, consoante o p-value.
  vars_significativas = ', '.join([
      col
      for col in resultado_modelo.pvalues.index
      if col != 'const' and resultado_modelo.pvalues[col] < 0.05
  ])
  vars_nao_significativas = ', '.join([
      col
      for col in resultado_modelo.pvalues.index
      if col != 'const' and resultado_modelo.pvalues[col] >= 0.05
  ])

  # Define o conteúdo textual com os principais resultados e interpretações do modelo.
  texto_interpretativo = f"""
====================================================================
 RELATÓRIO INTERPRETATIVO DO MODELO DE REGRESSÃO LOGÍSTICA
 CONTEXTO: PREVISÃO DE SOBREVIVÊNCIA NO TITANIC (ESTRATIFICADO)
====================================================================

1. RESUMO GERAL DO MODELO
   -----------------------------------------------------------------
   - Tipo de Modelo: Regressão Logística Binária (sm.Logit)
   - Divisão de Dados: train_test_split (stratify=y, test_size=0.2)
   - Total de Observações de Treino: {len(dados_treino)} passageiros
   - Total de Observações de Teste: {len(dados_teste)} passageiros
   - Qualidade do Ajuste (Pseudo R²): {pseudo_r2:.4f} (Melhoria significativa face ao modelo nulo)
   - Significância Global (LLR p-value): {llr_pvalue:.2e} (p < 0.001)

2. INTERPRETAÇÃO DOS COEFICIENTES E IMPORTÂNCIA
   -----------------------------------------------------------------
   - Sexo Masculino ('sex_male', p = {pvalue_sex:.3e}): Coeficiente ({coef_sex:.4f}).
     Indica que ser homem reduz substancialmente a probabilidade de
     sobrevivência em relação a ser mulher.
   - Classe ('pclass', p = {pvalue_pclass:.3e}): Coeficiente ({coef_pclass:.4f}).
     Passageiros em classes numéricas mais elevadas (ex. 3ª classe)
     apresentaram menor probabilidade de sobrevivência.
   - Idade ('age', p = {pvalue_age:.3e}): Coeficiente ({coef_age:.4f}).
     Idades mais avançadas reduziram ligeiramente a probabilidade.
   - Irmãos/Cônjuges ('sibsp', p = {pvalue_sibsp:.3f}): Coeficiente ({coef_sibsp:.4f}).
     Apresentou significância estatística negativa na sobrevivência.
   - Outras Variáveis ('parch', 'fare', dummies 'embarked'):
     Não apresentaram significância estatística ao nível de 5% (p > 0.05).

3. DESEMPENHO NO CONJUNTO DE TESTE
   -----------------------------------------------------------------
   - Taxa Global de Acerto (Acurácia): {taxa_acerto_global * 100:.2f}%
   - Verdadeiros Negativos (Mortes Previstas Corretamente): {matriz_confusao.iloc[0, 0]}
   - Verdadeiros Positivos (Sobrevivências Previstas Corretamente): {matriz_confusao.iloc[1, 1]}
   - Falsos Positivos (Previu Sobrevivência, mas Faleceu): {matriz_confusao.iloc[0, 1]}
   - Falsos Negativos (Previu Falecimento, mas Sobreviveu): {matriz_confusao.iloc[1, 0]}

4. NOTAS ESTATÍSTICAS E CONCLUSÃO
   -----------------------------------------------------------------
   - Pseudo R²: Mede o ganho de ajustamento face ao modelo nulo.
   - p-valor (p < 0.05): Indica evidência estatística para considerar
     o efeito significativo no modelo.
   - Conclusão: O modelo apresentou uma taxa global de acerto de {taxa_acerto_global * 100:.2f}%
     no conjunto de teste. As variáveis ({vars_significativas}) apresentaram
     significância estatística no modelo (p < 0.05), enquanto as variáveis
     ({vars_nao_significativas}) não apresentaram significância estatística ao nível
     de 5%. Os resultados indicam que as variáveis estatisticamente significativas
     apresentaram evidência de associação com a variável resposta no modelo.
"""

  # Escreve o texto interpretativo na página em branco criada anteriormente.
  ax_texto.text(
      0.05,
      0.95,
      texto_interpretativo,
      transform=ax_texto.transAxes,
      fontsize=9.0,
      fontfamily='monospace',
      verticalalignment='top',
  )

  # Cria uma segunda página do PDF com dois gráficos empilhados verticalmente.
  figura_graficos, (eixo_matriz, eixo_taxa_sobrevivencia) = plt.subplots(
      2, 1, figsize=(8.5, 11)
  )

  # Representa graficamente a matriz de confusão.
  sns.heatmap(
      matriz_confusao,
      annot=True,
      fmt='d',
      cmap='Blues',
      cbar=False,
      ax=eixo_matriz,
      annot_kws={'size': 14, 'weight': 'bold'},
  )

  eixo_matriz.set_title(
      '1. Matriz de Confusão (Desempenho no Conjunto de Teste - Estratificado)',
      fontsize=12,
  )
  eixo_matriz.set_xlabel('Previsão do Modelo (0 = Falecido, 1 = Sobreviveu)')
  eixo_matriz.set_ylabel('Realidade Observada (0 = Falecido, 1 = Sobreviveu)')

  plotar_taxa_sobrevivencia_por_classe_sexo(df_imputado, eixo_taxa_sobrevivencia)
  eixo_taxa_sobrevivencia.set_title(
      '2. Análise Exploratória: ' + eixo_taxa_sobrevivencia.get_title(),
      fontsize=12,
  )

  # Ajusta os espaçamentos entre os elementos do gráfico para evitar sobreposição.
  plt.tight_layout(pad=3.0)

  try:
    caminho_relatorio = BASE_DIR / 'relatorio_titanic.pdf'
    # Exporta o resumo e os gráficos para o relatório PDF.
    with PdfPages(caminho_relatorio) as gerador_pdf:
      gerador_pdf.savefig(figura_texto_resumo)
      gerador_pdf.savefig(figura_graficos)
    print(f"Relatório PDF exportado com sucesso para '{caminho_relatorio}'.")
  except PermissionError:
    print(
        "Aviso: Não foi possível gravar 'relatorio_titanic.pdf' porque o"
        ' ficheiro está aberto num visualizador. Feche o PDF e volte a'
        ' executar.'
    )

  # Fecha todas as figuras abertas para libertar memória.
  plt.close('all')


# Função principal que coordena todo o fluxo do projeto.
def main():
  caminho_ficheiro = BASE_DIR / 'titanic_v2.csv'
  df_titanic = carregar_dados(caminho_ficheiro)
  explorar_dados(df_titanic)
  explorar_dados_graficamente(df_titanic)
  df_dados, df_imputado = preparar_dados(df_titanic)

  (
      dados_treino,
      dados_teste,
      variavel_resposta_treino,
      variavel_resposta_teste,
      preditores_treino,
      preditores_teste,
  ) = dividir_dados(df_dados)

  resultado_modelo = treinar_modelo(variavel_resposta_treino, preditores_treino)
  print(resultado_modelo.summary())

  _, previsoes_binarias, matriz_confusao, taxa_acerto_global = avaliar_modelo(
      resultado_modelo, variavel_resposta_teste, preditores_teste
  )

  print(matriz_confusao)
  print(
      'Taxa de Acerto Global: {0}%'.format(np.round(taxa_acerto_global * 100, 2))
  )

  gerar_relatorio_pdf(
      dados_treino,
      dados_teste,
      matriz_confusao,
      taxa_acerto_global,
      resultado_modelo,
      df_imputado,
  )


# Executa o fluxo principal do projeto.
if __name__ == '__main__':
  main()