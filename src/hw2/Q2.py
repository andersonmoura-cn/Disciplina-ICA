import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import statsmodels.api as sm
import biblioteca as bib

np.random.seed(16)

# dados
base = Path(__file__).resolve().parent.parent.parent
arquivo_treino = base / "data" / "df_treino.csv"
df_treino = pd.read_csv(arquivo_treino)
arquivo_teste = base / "data" / "df_teste.csv"
df_teste = pd.read_csv(arquivo_teste)

X_train = df_treino.drop("HR", axis=1)
y_train = df_treino["HR"]

X_test = df_teste.drop("HR", axis=1)
y_test = df_teste["HR"]

#####################################################
print("Melhor lambda com nossa implementação: \n")
# Nossa implementacao
# escolhendo melhor lmbda
lambida = np.linspace(0, 1, num=10)
k = 5
metrics = [rmse, r2]

lamb_r2, lamb_rmse = bib.melhor_lambda(X_train, y_train, lambida, k, metrics, shuffle=False)

# treinando modelo com todo conjunto de dados
Beta, _ = bib.OLS_beta(X_train, y_train, lambida=lamb_rmse)
y_pred = bib.OLS_predict(X_test, Beta)

# avaliar modelo
print("Aplicando o melhor lambda na OLS de Ridge feita por nós: \n")
print(f"RMSE teste (k = {k}): {rmse(y_test, y_pred)}", end='\n\n')
print(f"R² teste (k = {k}): {r2(y_test, y_pred)}", end='\n\n')
print("-"*30)
# print(f"Y real de teste: {y_test}")
# print("------------------------------------------------")
# print(f"Y predizido: {y_pred}")

# Implementacao pronta
print("Aplicando o melhor lambda na OLS de Ridge built-in: \n")
X_tr_sm = sm.add_constant(X_train)
X_val_sm  = sm.add_constant(X_test)

model_cv = sm.OLS(y_train, X_tr_sm).fit_regularized(method='elastic_net', alpha=lamb_rmse, L1_wt=0)
y_pred_sm = model_cv.predict(X_val_sm)

print(f"RMSE teste (k = {k}) com função pronta: {rmse(y_test, y_pred_sm)}", end='\n\n')
print(f"R² teste (k = {k}) com função pronta: {r2(y_test, y_pred_sm)}", end='\n\n')

#####################################################
# CV pronto
print("Melhor lambda com KCV built-in: \n")

kf = KFold(n_splits=k, shuffle=False, random_state=None)

erroMedio_rmse = np.inf
erroMedio_r2 = -np.inf
lambd_rmse = lambida[0]
lambd_r2 = lambida[0]

for lmb in lambida:
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

        X_tr_sm = sm.add_constant(X_tr)
        X_val_sm  = sm.add_constant(X_val)

        model_cv = sm.OLS(y_tr, X_tr_sm).fit_regularized(method='elastic_net', alpha=lmb, L1_wt=0)
        y_pred_sm = model_cv.predict(X_val_sm)
        # adiciona constante em CADA fold
        # Beta, _ = bib.OLS_beta(X_tr, y_tr, lambida=lmb, pinv = False)
        # y_pred = bib.OLS_predict(X_val, Beta)

        rmse_scores.append(rmse(y_val, y_pred_sm))
        r2_scores.append(r2(y_val, y_pred_sm))

    # minimizar rmse
    if np.mean(rmse_scores) < erroMedio_rmse:
        erroMedio_rmse = np.mean(rmse_scores)
        lambd_rmse = lmb
    
    # maximizar r2
    if np.mean(r2_scores) > erroMedio_r2:
        erroMedio_r2 = np.mean(r2_scores)
        lamb_r2 = lmb

print(f"Para RMSE, Melhor lambda: {lambd_rmse}, com erro de {erroMedio_rmse}")
print(f"Para R², Melhor lambda: {lambd_r2}, com erro de {erroMedio_r2}") 


