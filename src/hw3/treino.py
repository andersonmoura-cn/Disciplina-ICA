import pandas as pd
from pathlib import Path

# dados
base = Path(__file__).resolve().parent.parent.parent
arquivo_treino = base / "data/hw3" / "df_treino.csv"
df_treino = pd.read_csv(arquivo_treino)
arquivo_teste = base / "data/hw3" / "df_teste.csv"
df_teste = pd.read_csv(arquivo_teste)

X_train = df_treino.drop("condition", axis=1)
y_train = df_treino["condition"]

X_test = df_teste.drop("condition", axis=1)
y_test = df_teste["condition"]

from sklearn.neighbors import NearestNeighbors
import numpy as np

nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(X_train)
dist, _ = nn.kneighbors(X_test)

print("min dist:", dist.min())
print("median dist:", np.median(dist))
print("pct dist < 1e-6:", (dist < 1e-6).mean())

