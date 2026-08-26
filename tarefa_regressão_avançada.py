# CONCEITOS AVANÇADOS DE REGRESSÃO

# OBJETIVO: Prever o valor do aluguel usando a variável metragem usando uma regressão polinomial

# 1. Importando bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # transformação das variáveis de entrada.
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score

# 2. Importando base de dados
pd.set_option("display.max_columns", None)
df_imoveis = pd.read_csv("ALUGUEL_MOD12.csv", delimiter=";")
print(df_imoveis)

# obs.: variável escolhida: 'Metragem'

y = df_imoveis['Valor_Aluguel']
x = df_imoveis[['Metragem']]

# 3. Processamento dos dados utilizando o Polynomial Feature, com degree = 2.
poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x)
print(x_poly)

# 4. Separando os dados em treino e teste e treinamento do modelo
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=0)

# Aplicando a regressão nos dados transformados
model = LinearRegression()
model.fit(x_train, y_train)

print("Tamanho de x_test: ", x_test.shape)
print("Tamanho de y_test: ", y_test.shape)

# 5. Fazendo as previsões para a base de teste e avaliando os resultados obtidos
y_pred = model.predict(x_test)

x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_poly = poly_features.transform(x_range)
y_range_pred = model.predict(x_range_poly)

plt.scatter(x, y, color='blue', label='Dados Reais')  # Dados reais
plt.plot(x_range, y_range_pred, color='green', label='Ajuste Polinomial')  # Linha do ajuste
plt.xlabel('Metragem')
plt.ylabel('Valor do Aluguel')
plt.title('Regressão Polinomial')
plt.legend()
plt.show()

# Avaliando valor de r2
r2 = r2_score(y_test, y_pred)
print(f'R2 da Regressão Polinomial: {r2:.4f}')
print("Mean squared error: ", mean_squared_error(y_test, y_pred))

# O valor de R2 foi de 0.5423, significa que o modelo consegue explicar 54% da variação da variável target a partir das
# variáveis preditoras. O restante fica sem explicação pelo modelo e podem ser resultado de outras variáveis não incluídas,
# ou da própria variabilidade dos dados.

# 6. Testando o processamento com o degree = 4.

poly_features = PolynomialFeatures(degree=4)

# Transformação da variável
x_poly = poly_features.fit_transform(x)
print(x_poly)

# Separando em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=0)

# Aplicando a regressão
model = LinearRegression()
model.fit(x_train, y_train)

print("Tamanho de x_test: ", x_test.shape)
print("Tamanho de y_test: ", y_test.shape)

print(poly_features.get_feature_names_out(['Metragem']))

# Fazendo as previsões
y_pred = model.predict(x_test)

# Plotando o gráfico
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_poly = poly_features.transform(x_range)
y_range_pred = model.predict(x_range_poly)

plt.scatter(x, y, color='blue', label='Dados Reais')  # Dados reais
plt.plot(x_range, y_range_pred, color='green', label='Ajuste Polinomial')  # Linha do ajuste
plt.xlabel('Metragem')
plt.ylabel('Valor do Aluguel')
plt.title('Regressão Polinomial')
plt.legend()
plt.show()

# Avaliando valor de r2
r2 = r2_score(y_test, y_pred)
print(f'R2 da Regressão Polinomial (degree = 4): {r2:.4f}')
print("Mean squared error (degree = 4): ", mean_squared_error(y_test, y_pred))

# 7. Avaliando e comparando os modelos

# O modelo de regressão polinomial com degree = 4 apresentou R2 = 0,3374, indicando que ele explica aproximadamente 33% da
# variação. Em comparação com o modelo anterior (degree = 2), que obteve R2 = 0,5423, houve uma redução no
# poder explicativo do modelo. Isso mostra que aumentar o grau do polinômio de 2 para 4 não melhorou o ajuste; pelo
# contrário, o modelo passou a representar pior a relação entre a metragem e o valor do aluguel nos dados de teste.

# O modelo com degree = 2 apresentou um MSE de 4.242.097, enquanto o modelo com degree = 4 obteve um MSE de 6.141.927.
# O aumento dessa métrica mostra que o modelo de grau 4 produziu previsões menos precisas. Esse resultado é consistente
# com a redução do R2, indicando que o aumento da complexidade do modelo não melhorou sua capacidade de generalização.
