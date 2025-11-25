import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split

np.random.seed(42)

def gerar_dados(n = 100, sigma = 0):
    x_teste = np.linspace(0, 10, n).reshape(-1, 1)
    ruido = np.random.normal(0, sigma, n)
    y_teste = 3 + 5 * x_teste[:, 0] + ruido
    return x_teste, y_teste

def OLS_beta(x, y, lambida = 0):
    x = np.array(x)
    y = np.array(y)
    beta_0 = np.ones((x.shape[0], 1))
    
    X = np.hstack((beta_0, x))

    if np.linalg.det(X.T @ X) == 0:
        print("Aviso: Matriz singular")
    else:
        inv = np.linalg.inv(X.T @ X + lambida * np.eye(X.shape[1]))
    Beta = inv @ X.T @ y 
    #print(X)
    return Beta, X

def OLS_predict(X, Beta):
    # ajuste para beta 0
    X_pred = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # predicao
    y_pred = X_pred @ Beta
    return y_pred

# x = [[1],[2],[3]]

# y = [[8],[10],[22]]

# Beta, X = OLS_beta(x, y)
# y_p = OLS_predict(X, Beta)
# print(y_p)


def k_fold(x,y,k):
    x = np.array(x)
    y = np.array(y)

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    X_emb = x[indices]
    Y_emb = y[indices]

    folds_indices = np.array_split(indices, k)

    folds = []
    for i in range(k):
        indices_teste = folds_indices[i]

        indices_treino = np.concatenate([folds_indices[j] for j in range(k) if j != i])

        X_treino, Y_treino = X_emb[indices_treino], Y_emb[indices_treino]
        X_teste, Y_teste = X_emb[indices_teste], Y_emb[indices_teste]

        folds.append([X_treino, Y_treino,X_teste, Y_teste])
    
    return folds


# x = [1,2,3,4,5,6,7,8,9,10]
# y = [3,4,5,2,3,7,7,8,9,10]

# lepo = k_fold(x, y, 5)
# for i in lepo:
#     print(i[0], i[1], i[2], i[3])

    
    


# Beta, X = OLS_beta(x, y)
# y_p = OLS_predict(X, Beta)
# print(y_p)


def k_fold(x,y,k):
    x = np.array(x)
    y = np.array(y)

    # embaralha indices
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    # embaralha dados usando indices
    X_emb = x[indices]
    Y_emb = y[indices]

    # divide indices em k partes
    folds_indices = np.array_split(indices, k)

    folds = []
    for i in range(k):
        indices_teste = folds_indices[i]

        # concatena indices de treino
        indices_treino = np.concatenate([folds_indices[j] for j in range(k) if j != i])

        # divisao dos dados
        X_treino, Y_treino = X_emb[indices_treino], Y_emb[indices_treino]
        
        X_teste, Y_teste = X_emb[indices_teste], Y_emb[indices_teste]
    
        folds.append([X_treino, Y_treino,X_teste, Y_teste])
    
    return folds


# x = [
#     [1, 69],
#     [2, 7],
#     [3, 2],
#     [4, 1],
#     [5, 9]
# ]

# y = [[3], [4],[5] , [2], [16]]

# lepo = k_fold(x, y, 5)

# for i in lepo:
#     print("Treino X:", i[0])
#     print("Treino Y:", i[1])
#     print("Teste  X:", i[2])
#     print("Teste  Y:", i[3])
#     print()

def kcv(x,y,params, k = 5, metric = rmse, modelo = OLS_beta):
    folds = k_fold(x, y, k)
    erros = []

    for lamb in params:
        erro_medio = 0
        n = 1
        for j in folds:
            # folds:  0: x_treino, 1: y_treino, 2: x_validacao, 3: y_validacao
            x_treino, y_treino, x_validacao, y_validacao = j[0], j[1], j[2], j[3]
            # escolha de beta
            Beta, _ = modelo(x_treino, y_treino, lamb)
            
            # predicao
            y_pred = OLS_predict(x_validacao, Beta)   

            # erro médio acumulado 
            erro_medio = erro_medio + metric(y_validacao, y_pred) / n
            n+=1
        
        erros.append(erro_medio)
    # indice do lambda que produziu menor erro medio
    indice_min = np.argmin(erros)
    return params[indice_min], erros
    
# teste
x_gerado, y_gerado = gerar_dados(n = 50, sigma = 1)
X_train, X_test, y_train, y_test = train_test_split(x_gerado, y_gerado, test_size=0.30)
lambida = [1,2,3]

ks = [5, 10]
metrics = [rmse, r2]

for metric in metrics:
    for k in ks:
        lamb, erros = kcv(X_train, y_train, lambida, k = k, metric=metric)
        print("Melhor lambda:", lamb)
        #print("Erros:", erros)

        # treina modelo com lambda escolhido
        Beta, _ = OLS_beta(X_train, y_train, lamb)

        # predicao
        y_pred = OLS_predict(X_test, Beta)  

        # avaliar modelo
        print(f"{metric.__name__} teste(k = {k}): {metric(y_test, y_pred)}")
        print()
  