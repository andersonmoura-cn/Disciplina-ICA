import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
from pathlib import Path
from biblioteca import rede_neural

base = Path(__file__).resolve().parent.parent.parent
arquivo_treino = base / "data" / "df_treino.csv"
df_treino = pd.read_csv(arquivo_treino)
arquivo_teste = base / "data" / "df_teste.csv"
df_teste = pd.read_csv(arquivo_teste)

X_train = df_treino.drop("HR", axis=1)
y_train = df_treino["HR"]

X_test = df_teste.drop("HR", axis=1)
y_test = df_teste["HR"]

# treinando rede neural com nosso dataset 
modelo = rede_neural()
history = modelo.fit(X_train, y_train, epochs = 100, batch_size = 256, validation_split = 0.2)

y_pred_test = modelo.predict(X_test)

r2_calc = r2(y_test,y_pred_test)
rmse_calc = rmse(y_test, y_pred_test)

print(f'RMSE ={rmse_calc} e R² ={r2_calc}')
print(history)
print(y_pred_test)