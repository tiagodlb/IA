####
#
# Exemplo de uma árvore de decisão treinada em pares de um dataset sobre as caracterísitcas da iris (Flor).
#
# Pra cada par de iris, a árvore de decisão aprende sobre os limites de combinações a partir das
# simples regras de inferidas pelas amostras de treino
# 
#
####

# Importa os módulos e livrarias necessárias
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# Dataset sobre iris
from sklearn.datasets import load_iris

# Carrega o dataset
iris = load_iris()
# Parametros
n_classes = 3
# Red Yellow Blue
plot_colors = "ryb"
# Passo
plot_step = 0.02

# Iteração a partir dos pares possíveis (3!)
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # Pegando somente as características correspondentes
    X = iris.data[:, pair]
    y = iris.target

    # Treinando a árvore de decisão
    clf = DecisionTreeClassifier().fit(X, y)

    # Plota o limite da decisão
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # Plota os pontos usados para o treino
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )
# Titulo e legenda
plt.suptitle("Árvore de decisão treinada em pares sobre as caracterísitcas da iris")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")
# Plota o gráfico
plt.show()