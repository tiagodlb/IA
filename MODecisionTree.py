####
#
# Exemplo de uma árvore de decisão com várias saidas.
# A árvore de decisão é usada para prever o ruído em x e em y, simultaneamente, de um círculo básico.
# Como resultado a árvore de decisão vai aprender regressão linear local para aproximar a curva. 
# 
# É perceptivel também, ver que caso max_depth (Profundidade) seja muito alto a árvore de decisão aprende muitos detalhes
# e isso acaba fazendo com que ela aprenda com os ruidos.
#
# Contudo, nesse caso, isso acabou sendo benéfico, uma vez que a arvore acabou aprendendo mais e conseguiu
# "fechar" o círclo.
#
####

# Importa os módulos e livrarias necessárias
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor

# Cria um dataset aleatório
rng = np.random.RandomState(1) # Random Number Generator
array2 = 300 * rng.rand(100,1) - 120  # Cria um array aleatório
X = np.sort(array2, axis=0) # Cópia do array1 organizado e "achatado" utilizando quicksort
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
#  y[inicia em: termina em : passo de]
y[::5, :] += 0.5 - rng.rand(20, 2) 

# Ajusta o modelo de regressão
regr_1 = DecisionTreeRegressor(max_depth=2) # Cria o objeto da árvore de regressão de profundidade 2
regr_2 = DecisionTreeRegressor(max_depth=5) # Cria o objeto da árvore de regressão de profundidade 5
regr_3 = DecisionTreeRegressor(max_depth=8) # Cria o objeto da árvore de regressão de profundidade 8
regr_1.fit(X, y) # adiciona os valores ao modelo 1
regr_2.fit(X, y) # adiciona os valores ao modelo 2
regr_3.fit(X, y) # adiciona os valores ao modelo 3

# Predição
X_test = np.arange(-100.0, 250.0, 0.02)[:, np.newaxis] # Cria um array de intervalos igualmente espaçados
y_1 = regr_1.predict(X_test) # Utiliza o modelo 1 para "prever" o resultado
y_2 = regr_2.predict(X_test) # Utiliza o modelo 2 para "prever" o resultados
y_3 = regr_3.predict(X_test) # Utiliza o modelo 3 para "prever" o resultados

# Plota os resultados
plt.figure() # Cria a figura
plt.scatter(y[:, 0], y[:, 1], c="navy", s=25, edgecolor="black", label="data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=25, edgecolor="black",label="Profundidade 2")
plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=25, edgecolor="black", label="Profundidade 5")
plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=25, edgecolor="black", label="Profundidade 8")
plt.xlim([-6, 6]) # Limita o eixo X 
plt.ylim([-6, 6]) # Limita o eixo Y
plt.xlabel("alvo 1") # Adiciona a legenda no eixo X
plt.ylabel("alvo 2") # Adiciona a legenda no eixo Y
plt.title("Arvore de Regressão de múltiplas saídas") # Adiciona o título do gráfico
plt.legend(loc="best") # Plota a legenda
plt.show() # Mostra o gráfico