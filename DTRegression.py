####
#
# Exemplo de uma árvore de decisão unidimensional.
# A árvore de decisão é usada para ajustar a curva de uma função seno. Adicionalmente temos ruídos.
# Como resultado a árvore de decisão vai aprender regressão linear local para aproximar a curva. 
#
####

# Importa os módulos e livrarias necessárias
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor

# Cria um dataset aleatório
rng = np.random.Generator(1) # Random Number Generator
array1 = 5 * rng.rand(80,1)
X = np.sort(array1, axis=0)   # Cópia do array1 organizado e "achatado" utilizando quicksort
y = np.sin(X).ravel() # cria uma função seno com base nos valores de X e achata
#  y[inicia em: termina em : passo de] #
y[::5] += 3 * (0.5 - rng.rand(16))  # a cada 5 valores, modifica o array utilizando essa "regra"

# Modelo de regressão
regr_1 = DecisionTreeRegressor(max_depth=2) # Cria o objeto da árvore de regressão
regr_2 = DecisionTreeRegressor(max_depth=5) # Cria uma outra árvore de regressão
regr_1.fit(X, y) # adiciona os valores ao modelo 1
regr_2.fit(X, y) # adiciona os valores ao modelo 2

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] # cria um array de intervalos igualmente espaçados
y_1 = regr_1.predict(X_test) # Utiliza o modelo 1 para "prever" o resultado
y_2 = regr_2.predict(X_test) # Utiliza o modelo 2 para "prever" o resultados

# Plota os resultados
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()