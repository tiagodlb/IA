####
#
# Exemplo de uma árvore de decisão unidimensional.
# A árvore de decisão é usada para ajustar a curva de uma função seno. Adicionalmente temos ruídos.
# Como resultado a árvore de decisão vai aprender regressão linear local para aproximar a curva. 
# 
# É perceptivel também, ver que caso max_depth (Profundidade) seja muito alto a árvore de decisão aprende muitos detalhes
# e isso acaba fazendo com que ela aprenda com os ruidos.
#
####

# Importa os módulos e livrarias necessárias
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.tree import DecisionTreeRegressor

# Cria um dataset aleatório
rng = np.random.RandomState(1) # Random Number Generator
array1 = 10 * rng.rand(80,1) # Cria um array aleatorio   
X = np.sort(array1, axis=0) # Cópia do array1 organizado e "achatado" utilizando quicksort
y = np.sin(X).ravel() # cria uma função seno com base nos valores de X e achata
#  y[inicia em: termina em : passo de] #
y[::5] += 3 * (0.5 - rng.rand(16))  # a cada 5 valores, modifica o array utilizando essa "regra"

# Modelo de regressão
regr_1 = DecisionTreeRegressor(max_depth=2) # Cria o objeto da árvore de regressão de profundidade 2
regr_2 = DecisionTreeRegressor(max_depth=5) # Cria o objeto da árvore de regressão de profundidade 5
regr_3 = DecisionTreeRegressor(max_depth=8) # Cria o objeto da árvore de regressão de profundidade 8
regr_1.fit(X, y) # adiciona os valores ao modelo 1
regr_2.fit(X, y) # adiciona os valores ao modelo 2
regr_3.fit(X, y) # adiciona os valores ao modelo 3 

# Predição
X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis] # Cria um array de intervalos igualmente espaçados
y_1 = regr_1.predict(X_test) # Utiliza o modelo 1 para "prever" o resultado
y_2 = regr_2.predict(X_test) # Utiliza o modelo 2 para "prever" o resultados
y_3 = regr_3.predict(X_test) # Utiliza o modelo 3 para "prever" os resultados

# Plota os resultados
plt.figure() # Cria a figura
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data") 
plt.plot(X_test, y_1, color="cornflowerblue", label="Profundidade 2", linewidth=2) # Plota o valor do modelo 1
plt.plot(X_test, y_2, color="indianred", label="Profundidade 5", linewidth=2) # Plota o valor do modelo 2
plt.plot(X_test, y_3, color="lightgreen", label="Profundidade 8", linewidth=2) # Plota o valor do modelo 3
plt.xlabel("dados") # Adiciona uma legenda para o eixo X
plt.ylabel("alvo") # Adiciona uma legenda pra o eixo Y
plt.title("Arvore de Regressão") # Adiciona um título para o gráfico
plt.legend() # Adiciona a legenda do matplot nos eixos 
plt.show() # Mostra o gráfico na tela