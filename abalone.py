from sklearn.linear_model import LinearRegression
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Fetch dataset
abalone = fetch_ucirepo(id=1)

# Data (as pandas dataframes)
X = abalone.data.features
y = abalone.data.targets['Rings']  # 'Rings' é a variável alvo que representa a idade

# Verificar a estrutura dos dados
print("X columns:", X.columns)
print("y head:", y.head())

# Verificar se a coluna 'Rings' está presente em X
if 'Rings' in X.columns:
    X = X.drop(columns=['Rings'])

# Convertendo colunas categóricas para numéricas se necessário (por exemplo, a coluna 'Sex')
X = pd.get_dummies(X, drop_first=True)

# Verificar tipos de dados após conversão
print(X.dtypes)

# Identificar e remover valores inválidos (NaNs, infinitos)
# Substituir valores infinitos por NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
# Verificar se ainda há NaNs
print("Número de NaNs em X antes de dropna:", X.isna().sum().sum())
# Remover NaNs
X.dropna(inplace=True)
# Garantir que y tenha o mesmo índice que X
y = y.loc[X.index]

# Verificar se ainda há NaNs em y
print("Número de NaNs em y antes de dropna:", y.isna().sum().sum())
# Remover NaNs de y
y.dropna(inplace=True)

# Verificar se ainda há valores inválidos
print("Número de NaNs em X após limpeza:", X.isna().sum().sum())
print("Número de NaNs em y após limpeza:", y.isna().sum().sum())

# Verificar se há valores infinitos
print("Número de valores infinitos em X após limpeza:", np.isinf(X).sum().sum())
print("Número de valores infinitos em y após limpeza:", np.isinf(y).sum().sum())

# Padronizando os dados para que todas as variáveis tenham média 0 e desvio padrão 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Criando o modelo de regressão de árvore de decisão
regressor = DecisionTreeRegressor(random_state=42)

# Definindo a grade de parâmetros para o GridSearchCV
param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicializando o GridSearchCV com validação cruzada de 5 vezes
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Ajustando o GridSearchCV aos dados de treinamento
grid_search.fit(X_train, y_train)

# Melhor conjunto de hiperparâmetros
print(f'Melhores hiperparâmetros: {grid_search.best_params_}')

# Treinando o modelo com os melhores hiperparâmetros
best_regressor = grid_search.best_estimator_

# Fazendo previsões com o melhor modelo
y_pred_best = best_regressor.predict(X_test)

# Avaliando o melhor modelo
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f'Melhor Erro Quadrático Médio: {mse_best:.2f}')
print(f'Melhor Coeficiente de Determinação R^2: {r2_best:.2f}')
new_y = y_test.reset_index()
fig, ax = plt.subplots()
reg = LinearRegression().fit(new_y, y_pred_best)
y_pred = reg.predict(new_y)
ax.scatter(y_test, y_pred_best,s=10, color="darkslateblue", linewidths=1)
ax.plot(y_test, y_pred , color="red", lw=1)
plt.show()