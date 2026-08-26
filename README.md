# regressao-polinomial-aluguel
Projeto desenvolvido como atividade prática de Machine Learning para aplicar conceitos avançados de regressão utilizando Regressão Polinomial com PolynomialFeatures do Scikit-learn.
# Regressão Polinomial: Previsão do Valor de Aluguel

Projeto desenvolvido como atividade prática de Machine Learning para aplicar conceitos avançados de regressão utilizando **Regressão Polinomial** com `PolynomialFeatures` do Scikit-learn.

## Objetivo

Construir um modelo capaz de prever o **valor do aluguel** a partir da variável **metragem**, comparando o desempenho de regressões polinomiais de diferentes graus.

## Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Etapas do projeto

* Importação e exploração da base de dados `ALUGUEL_MOD12.csv`;
* Definição da variável preditora (`Metragem`) e da variável alvo (`Valor_Aluguel`);
* Transformação da variável de entrada com `PolynomialFeatures`;
* Divisão dos dados em treino e teste (`train_test_split`);
* Treinamento da regressão polinomial com:

  * `degree = 2`
  * `degree = 4`
* Geração dos gráficos de ajuste dos modelos;
* Avaliação utilizando as métricas **R²** e **Mean Squared Error (MSE)**.

## Comparação dos modelos

| Modelo     |     R² |       MSE |
| ---------- | -----: | --------: |
| Degree = 2 | 0.5423 | 4,242,097 |
| Degree = 4 | 0.3374 | 6,141,927 |

## Principais resultados

* O modelo com **degree = 2** explicou aproximadamente **54% da variação** do valor dos aluguéis.
* O aumento do grau para **4** reduziu o desempenho do modelo, diminuindo o R² para **33%**.
* O MSE também aumentou, indicando previsões menos precisas.
* Os resultados mostram que um modelo mais complexo nem sempre apresenta melhor capacidade de generalização, podendo reduzir o desempenho em dados de teste.

## Estrutura do repositório

```text
├── tarefa_regressão_avançada.py
├── ALUGUEL_MOD12.csv
└── README.md
```

## Conceitos aplicados

* Regressão Polinomial
* Engenharia de atributos com `PolynomialFeatures`
* Divisão treino e teste
* Avaliação de modelos de regressão
* Interpretação de R²
* Mean Squared Error (MSE)
* Comparação de modelos e capacidade de generalização
