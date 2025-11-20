import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

x = [[1],[2],[3]]

y = [[8],[10],[22]]

Beta, X = OLS_beta(x, y)
y_p = OLS_predict(X, Beta)
print(y_p)



