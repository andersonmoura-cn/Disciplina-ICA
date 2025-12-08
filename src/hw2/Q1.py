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

#############################################################
# Comparação do nosso modelo de OLS com a função pronta
Beta, _ = bib.OLS_beta(X_train, y_train)
y_pred = bib.OLS_predict(X_test, Beta)

#funcao pronta

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Treinamento
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Predição
y_pred_sm = model_sm.predict(X_test_sm)

# Métricas
rmse_sm = rmse(y_test, y_pred_sm)
r2_sm = r2(y_test, y_pred_sm)

# COMPARAÇÃO PARTE 1
print(" ")
print("Aplicação da OLS")
print(" ")

print(f"{'Métrica'} | {'Funcao do Zero(1)'} | {'Funcao Pronta(1)'}")


print(f"{'RMSE'} | {rmse(y_test, y_pred)} | {rmse_sm}")
print(f"{'R2'} | {r2(y_test, y_pred)} | {r2_sm}")

# Comparação dos erros obtidos na cross validation
k = 5
rmse_scores = []
r2_scores = []

print(" ")
print("Erro médio aplicando cross validation com k = 5")
print(" ")
# Funcao criada

erro_rmse = bib.kcv(X_train, y_train, k = k, metric=rmse, shuffle=False)
erro_r2 = bib.kcv(X_train, y_train, k = k, metric=r2, shuffle=False)

print(f"RMSE médio (CV {k} folds) func do zero: {erro_rmse:.4f}")
print(f"R² médio   (CV {k} folds) func do zero: {erro_r2:.4f}")
print("---------------------------------------------------------")

# Funcao pronta

kf = KFold(n_splits=k, shuffle=False, random_state=None)

for train_idx, test_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

    # adiciona constante em CADA fold
    X_tr_sm = sm.add_constant(X_tr)
    X_val_sm  = sm.add_constant(X_val)

    model_cv = sm.OLS(y_tr, X_tr_sm).fit()
    y_pred_cv = model_cv.predict(X_val_sm)

    rmse_scores.append(rmse(y_val, y_pred_cv))
    r2_scores.append(r2(y_val, y_pred_cv))

print(f"RMSE médio (CV {k} folds) func pronta: {np.mean(rmse_scores):.4f}")
print(f"R² médio   (CV {k} folds) func pronta: {np.mean(r2_scores):.4f}")
