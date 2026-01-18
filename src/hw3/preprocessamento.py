import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder,  StandardScaler


def skew_fit_transform(df, colunas):
    dados_transformados = pd.DataFrame()
    info_transf = {}  # guarda método e parâmetros por coluna
    
    for col in colunas:
        data = df[col].astype(float)

        # 1. Skew Original
        original_skew = data.skew()

        # Dicionário temporário para guardar skews desta variável
        metodos = {'Original': abs(original_skew)}
        series_temp = {'Original': data} # Para guardar as séries transformadas

        # 2. Logaritmo
        if data.min() >= 0:
            log_data = np.log1p(data)
            metodos['Log'] = abs(log_data.skew())
            series_temp['Log'] = log_data
        else:
            metodos['Log'] = 999 # Penalidade se não puder aplicar

        # 3. Box-Cox
        try:
            # Se tiver 0 ou negativo, deslocamos os dados
            shift = 0
            if data.min() <= 0:
                shift = abs(data.min()) + 1

            boxcox_data, lmbda = stats.boxcox(data + shift)
            boxcox_data = pd.Series(boxcox_data, index=data.index)
            metodos['Box-Cox'] = abs(boxcox_data.skew())
            series_temp['Box-Cox'] = boxcox_data
        except:
            shift = None
            lmbda = None
            metodos['Box-Cox'] = 999

        # 4. Yeo-Johnson (Funciona com negativos)
        try:
            yeo_data, lmbda_yj  = stats.yeojohnson(data)
            yeo_data = pd.Series(yeo_data, index=data.index)
            metodos['Yeo-Johnson'] = abs(yeo_data.skew())
            series_temp['Yeo-Johnson'] = yeo_data
        except:
            lmbda_yj = None
            metodos['Yeo-Johnson'] = 999

        # 5. Tangente 
        try:
            tan_data = np.tan(data)
            # Tangente pode gerar valores absurdos, então verificamos se o skew não é NaN/Inf
            if np.isfinite(tan_data).all():
                sk_tan = tan_data.skew()
                if np.isfinite(sk_tan):
                    metodos['Tangente'] = abs(sk_tan)
                    series_temp['Tangente'] = tan_data
                else:
                    metodos['Tangente'] = 999
            else:
                metodos['Tangente'] = 999
        except:
            metodos['Tangente'] = 999

        # 6. Raiz Quadrada (Adicionar este bloco no loop de comparação)
        try:
            shift_raiz = 0
            min_val = data.min()
            # Se houver negativos, precisamos de shift para a raiz
            if min_val < 0:
                shift_raiz = abs(min_val)
            
            raiz_data = np.sqrt(data + shift_raiz)
            sk_raiz = abs(raiz_data.skew())
            
            if np.isfinite(sk_raiz):
                metodos['Raiz'] = sk_raiz
                series_temp['Raiz'] = raiz_data
                # Precisamos salvar o shift para usar depois na função skew_transform
                # Obs: Você precisará adaptar sua estrutura de salvamento para guardar o shift da Raiz também
            else:
                metodos['Raiz'] = 999
        except:
            metodos['Raiz'] = 999

        # --- COMPARAÇÃO ---

        # método com o skew absoluto mais próximo de 0
        melhor_metodo = min(metodos, key=metodos.get)
        dados_transformados[col] = series_temp[melhor_metodo]

        # info pra usar no transform
        cfg = {'method': melhor_metodo}
        if melhor_metodo == 'Box-Cox':
            cfg['shift'] = shift
            cfg['lambda'] = lmbda
        elif melhor_metodo == 'Yeo-Johnson':
            cfg['lambda'] = lmbda_yj
        elif melhor_metodo == 'Raiz':
            cfg['shift'] = shift_raiz

        info_transf[col] = cfg

    return dados_transformados, info_transf

def skew_transform(df, colunas, info):
    dados_transformados = pd.DataFrame(index=df.index)

    for col in colunas:
        cfg = info[col]

        metodo = cfg['method']
        data = df[col].astype(float)

        #Foi recomendado colocar isso:
        data = df[col].astype(float)

        if metodo == 'Original':
            dados_transformados[col] = data

        elif metodo == 'Raiz':
            shift = cfg.get('shift', 0)
            dados_transformados[col] = np.sqrt(data + shift)    
        
        elif metodo == 'Log':
            dados_transformados[col] = np.log1p(data)

        elif metodo == 'Box-Cox':
            shift = cfg['shift']
            lmbda = cfg['lambda']
            dados_transformados[col] = stats.boxcox(data + shift, lmbda=lmbda)

        elif metodo == 'Yeo-Johnson':
            lmbda = cfg['lambda']
            dados_transformados[col] = stats.yeojohnson(data, lmbda=lmbda)

        elif metodo == 'Tangente':
            dados_transformados[col] = np.tan(data)

        else:
            raise ValueError(f"Método desconhecido: {metodo}")

    return dados_transformados


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
arquivo_treino = base / "data" / "dados_nao_tratados.csv"
df_M = pd.read_csv(arquivo_treino)

# removendo 'time pressure'
df_M = df_M.query("condition != 'time pressure'").reset_index(drop=True)

# preditores filtrados
X = df_M[variaveis].copy()
y = df_M["condition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#########################################################################
# Ajustar transformações e aplicar em treino
X_train_proc, info = skew_fit_transform(X_train[variaveis], variaveis)
print(info)
# Padronização
scaler = StandardScaler()
X_train_scaled_np = scaler.fit_transform(X_train_proc)

X_train_scaled = pd.DataFrame(X_train_scaled_np, 
                               columns=X_train_proc.columns, 
                               index=X_train_proc.index)

# Cria as colunas numéricas
encoder = LabelEncoder()
y_train_encoder = encoder.fit_transform(y_train)
# verificar
for label, classe in enumerate(encoder.classes_):
    print(f"{classe} -> {label}")

# dataframe treino final
train_df = X_train_scaled.assign(condition=y_train_encoder)

#########################################################################
# Aplicar transformações no teste

# 1) Mesma transformação de skew (usa info do treino)
X_test_proc = skew_transform(X_test[variaveis], variaveis, info)

# 2) Padronização (usa scaler ajustado no treino)
X_test_scaled_np = scaler.transform(X_test_proc)

X_test_scaled = pd.DataFrame(
    X_test_scaled_np,
    columns=X_test_proc.columns,
    index=X_test_proc.index
)

# 3) Encode do target (usa encoder do treino)
y_test_encoder = encoder.transform(y_test)

# dataframe teste final
test_df = X_test_scaled.assign(condition=y_test_encoder)

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




