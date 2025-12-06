import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from pathlib import Path
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

#####################################################
p = X_train.shape[1]    
Ms = np.arange(1, (p + 1), 1)
k = 5
metrics = [rmse, r2]

print("Melhor M para PLS:")
m_RMSE_PLS, m_R2_PLS = bib.melhor_M(X_train, y_train, k, Ms, 1, metrics)

print("Melhor M para PCR:")
m_RMSE_PCR, m_R2_PCR = bib.melhor_M(X_train, y_train, k, Ms, 2, metrics)

print("-----------------------------------PLS------------------------------------")
# PLS
Z_train, parameters = bib.PLS_fit_transform(X_train, y_train, m_RMSE_PLS)
Z_test = bib.PLS_transform(X_test, parameters)

# nossa implementacao
# escolha de beta 
Beta, _ = bib.OLS_beta(Z_train, y_train)
                
# predicao
y_predicted = bib.OLS_predict(Z_test, Beta)

 # avaliar modelo
print(f"RMSE teste(k = {k}): {rmse(y_test, y_predicted)}")
print()
print(f"R² teste(k = {k}): {r2(y_test, y_predicted)}")
print("---------------------------")
print(f"Y real de teste: {y_test}")
print(f"Y predizido: {y_predicted}")

print("-----------------------------------PCR------------------------------------")
 #PCR
vecs, vals = bib.PCA(X_train)
Z_train = bib.PCA_transform(X_train, vecs, n_componentes=m_RMSE_PCR)
Z_test = bib.PCA_transform(X_test, vecs, n_componentes=m_RMSE_PCR)

# nossa implementacao
# escolha de beta 
Beta, _ = bib.OLS_beta(Z_train, y_train)
                
# predicao
y_predicted = bib.OLS_predict(Z_test, Beta)

 # avaliar modelo
print(f"RMSE teste(k = {k}): {rmse(y_test, y_predicted)}")
print()
print(f"R² teste(k = {k}): {r2(y_test, y_predicted)}")
print("---------------------------")
print(f"Y real de teste: {y_test}")
print(f"Y predizido: {y_predicted}")