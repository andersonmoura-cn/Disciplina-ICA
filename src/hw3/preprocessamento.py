import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,  StandardScaler


np.random.seed(16)

# variaveis = [
#     "TP",
#     "SD2",
#     "HF",
#     "LF",
#     "VLF_PCT",
#     "MEDIAN_RR",
#     "RMSSD",
#     "SDRR_RMSSD_REL_RR",
#     "HF_LF",
#     "SDRR_REL_RR",
#     "SDSD_REL_RR",
#     "HR"
# ]

variaveis = [
    "TP",
    "SD2",
    "HF",
    "LF",
    "VLF_PCT",
    "HF_LF"
]

# pegando dados não tratados
base = Path.cwd().parents[1]
arquivo_treino = base / "data" / "original" / "dados_nao_tratados.csv"
df_M = pd.read_csv(arquivo_treino)

# removendo 'time pressure'
df_M = df_M.query("condition != 'time pressure'").reset_index(drop=True)

# preditores filtrados
X = df_M[variaveis].copy()
y = df_M["condition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

#########################################################################
# encode do target
encoder = LabelEncoder()
y_train_encoder = encoder.fit_transform(y_train)
# verificar
for label, classe in enumerate(encoder.classes_):
    print(f"{classe} -> {label}")

# dataframe treino final
train_df = X_train.assign(condition=y_train_encoder)

#########################################################################
# Aplicar transformações no teste

# Encode do target (usa encoder do treino)
y_test_encoder = encoder.transform(y_test)

# dataframe teste final
test_df = X_test.assign(condition=y_test_encoder)

##########################################################################
# Salvando

def save(data, name):
    # Caminho base: projeto (DISCIPLINA-ICA)
    base = Path(__file__).resolve().parent.parent.parent  

    # Caminho da pasta de imagens
    folder = base / "data/hw3"
    folder.mkdir(parents=True, exist_ok=True)

    # Caminho completo do arquivo
    file_path = folder / f"{name}.csv"

    data.to_csv(file_path, index=False)


# matriz tratada
save(train_df, 'df_treino')
print("Arquivo 'df_treino.csv' salvo com sucesso! :)")
save(test_df, 'df_teste')
print("Arquivo 'df_teste.csv' salvo com sucesso! :)")

###################################################333333




