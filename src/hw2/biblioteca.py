import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse

def OLS_beta(x, y):
    x = np.array(x)
    y = np.array(y)
    beta_0 = np.ones((x.shape[0], 1))
    X = np.hstack((beta_0, x))

    if np.linalg.det(X.T @ X) == 0:
        print("Aviso: Matriz singular")
    else:
        inv = np.linalg.inv(X.T @ X)
    Beta = inv @ X.T @ y
    print(X)
    return Beta, X

def OLS_predict(X, Beta):
    y_pred = X @ Beta
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


x = [1,2,3,4,5,6,7,8,9,10]
y = [3,4,5,2,3,7,7,8,9,10]

lepo = k_fold(x, y, 5)
for i in lepo:
    print(i[0], i[1], i[2], i[3])

    
    




