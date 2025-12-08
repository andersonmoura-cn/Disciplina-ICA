import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from tensorflow import keras 
# from tensorflow.keras import layers
import keras
from keras import layers

###########################################################
# seed
np.random.seed(16)

###########################################################
# salvar imagem
def img_save(file_name):
    # Caminho base: projeto (DISCIPLINA-ICA)
    base = Path(__file__).resolve().parent.parent.parent  

    # Caminho da pasta de imagens
    folder = base / "images" / "hw2"
    folder.mkdir(parents=True, exist_ok=True)

    # Caminho completo do arquivo
    file_path = folder / f"{file_name}.png"

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Imagem salva em: {file_path}")

###########################################################
# OLS ajuste e transformacao
def OLS_beta(x, y, lambida = 0, pinv = True):
    x = np.array(x)
    y = np.array(y)
    beta_0 = np.ones((x.shape[0], 1))
    
    X = np.hstack((beta_0, x))

    # if np.linalg.det(X.T @ X + lambida * np.eye(X.shape[1])) == 0:
    #     print("Aviso: Matriz singular")
    #     inv = np.linalg.pinv(X.T @ X + lambida * np.eye(X.shape[1]))
    # else:
    #     inv = np.linalg.inv(X.T @ X + lambida * np.eye(X.shape[1]))

    

    #print("Tem NaNs no X?", np.isnan(X).any() if hasattr(X, 'any') else "N/A")
    #print("-------------")

    # Beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    
    if pinv:
        inv = np.linalg.pinv(X.T @ X + lambida * np.eye(X.shape[1]))
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

###########################################################
# k-fold
def k_fold(x,y,k, shuffle = True):
    x = np.array(x)
    y = np.array(y)

    # embaralha indices
    indices = np.arange(len(x))

    if shuffle:
        np.random.shuffle(indices)

    # # embaralha dados usando indices
    # X_emb = x[indices]
    # Y_emb = y[indices]

    # divide indices em k partes
    folds_indices = np.array_split(indices, k)

    folds = []
    for i in range(k):
        indices_teste = folds_indices[i]

        # concatena indices de treino
        indices_treino = np.concatenate([folds_indices[j] for j in range(k) if j != i])

        # divisao dos dados
        X_treino, Y_treino = x[indices_treino], y[indices_treino]
        
        X_teste, Y_teste = x[indices_teste], y[indices_teste]
    
        folds.append([X_treino, Y_treino,X_teste, Y_teste])
    
    return folds

###########################################################
# kcv
def kcv(x, y, lamb = 0, k = 5, metric = rmse, modelo = OLS_beta, shuffle = True, pinv = True):
    folds = k_fold(x, y, k, shuffle=shuffle)
    erro_medio = 0
    for j in folds:
        # folds:  0: x_treino, 1: y_treino, 2: x_validacao, 3: y_validacao
        x_treino, y_treino, x_validacao, y_validacao = j[0], j[1], j[2], j[3]
        # escolha de beta
        Beta, _ = modelo(x_treino, y_treino, lamb, pinv = pinv)
        
        # predicao
        y_pred = OLS_predict(x_validacao, Beta)   

        y_validacao = y_validacao.ravel()
        y_pred = y_pred.ravel()

        # erro médio acumulado 
        #erro_medio = erro_medio + metric(y_validacao, y_pred) / n
        erro_medio += metric(y_validacao, y_pred)

    
    erro_medio /= len(folds)
    return erro_medio


###########################################################
# escolha de melhor parametro lambda
def melhor_lambda(x, y, lambida, k, metrics, modelo = OLS_beta, shuffle = True):
    erro_rmse = 0
    erro_r2 = 0
    lambd_r2 = None
    lambd_rmse = None
    for metric in metrics:
        erros = []
        for lamb in lambida:
            erro_medio = kcv(x, y, lamb = lamb, k = k, metric = metric, modelo = modelo, shuffle=shuffle, pinv = False)
            erros.append(erro_medio)
         
        if metric == r2:
            lambd_r2 = lambida[np.argmax(erros)]
            erro_r2 = max(erros)
            #print(erros)
        else:
            lambd_rmse = lambida[np.argmin(erros)]
            erro_rmse = min(erros)
            #print(erros)

        
    print(f"Para RMSE, Melhor lambda: {lambd_rmse}, com erro de {erro_rmse}")
    print(f"Para R², Melhor lambda: {lambd_r2}, com erro de {erro_r2}")    
    return lambd_r2, lambd_rmse

###########################################################
# escolha de melhor parametro M

def melhor_M(X_train, y_train, k, Ms, modelo, metrics): 
    # modelo 1 = PLS; modelo 2 = PCR
    folds = k_fold(X_train, y_train, k, shuffle=False)
    erro_rmse = 0
    erro_r2 = 0
    for metric in metrics:
        erros = []
        for M in Ms:
            erro_metrica = []
            erro_medio = 0
            for j in folds:
                # folds:  0: x_treino, 1: y_treino, 2: x_validacao, 3: y_validacao
                x_treino, y_treino, x_validacao, y_validacao = j[0], j[1], j[2], j[3]
                
                if modelo == 1:
                    # PLS
                    Z_treino, parametros = PLS_fit_transform(x_treino, y_treino, M)
                    Z_teste = PLS_transform(x_validacao, parametros)
                else:
                    #PCR
                    vecs, vals = PCA(x_treino)
                    Z_treino = PCA_transform(x_treino, vecs, n_componentes=M)
                    Z_teste = PCA_transform(x_validacao, vecs, n_componentes=M)
                    
                # nossa implementacao
                # escolha de beta 
                Beta, _ = OLS_beta(Z_treino, y_treino)
                
                # predicao
                y_pred = OLS_predict(Z_teste, Beta)   

                # erro médio acumulado 
                erro_metrica.append(metric(y_validacao, y_pred))
                
            erro_medio = np.mean(erro_metrica)
            erros.append(erro_medio)

            if len(erros) > 1 and (np.abs(erros[-1] - erros[-2]) < 1e-2):
                break


        if metric == r2:
            m_R2 = Ms[np.argmax(erros)]
            erro_r2 = max(erros)
            #print(erros)
        else:
            m_RMSE = Ms[np.argmin(erros)]
            erro_rmse = min(erros)
            #print(erros)

    print(f"Para RMSE, Melhor M: {m_RMSE}, com erro de {erro_rmse}")
    # print(f"Para R2, Melhor M: {m_R2}, com erro de {erro_r2}")
        
    return m_RMSE, m_R2
    

##########################################################
# PCR ajuste e transformacao

# pre-requisito: Z = X padronizado
def PCA(Z):
    # matriz de covariancia
    Cov = pd.DataFrame(Z).cov()

    # autovetores e autovalores
    vals, vecs = np.linalg.eigh(Cov)              # C é simétrica
    order = np.argsort(vals)[::-1]              # ordem decrescente
    vals = vals[order]
    vecs = vecs[:, order]
    
    return vecs, vals

def PCA_transform(Z, vecs, n_componentes):
    Z_pca = Z @ vecs[:, :n_componentes]
    return Z_pca


##########################################################
# PLS ajuste e transformacao

# Pré-requisito: x padronizado 
def PLS_fit_transform(x, Y, M):
    x = np.asarray(x).copy()
    Y = np.asarray(Y)

    # hipoteses
    y_mean = np.mean(Y)
    y_cartola = Y - y_mean

    # código
    y_chapeu = np.full_like(y_cartola, y_cartola.mean())

    # parametros
    theta_list = []
    phi_list = []
    load_list = []
    Z_list = []
    
    for m in range(M):
        Zm = np.zeros_like(y_cartola)
        phi_m = []

        # pesos
        for j in range(x.shape[1]):
            phi_mj = np.dot(x[:, j],y_cartola)
            phi_m.append(phi_mj)
            Zm += phi_mj * x[:, j]
            
        # if np.dot(Zm, Zm) == 0:
        #     print(f"Zm: {Zm} \n m|M: {m}|{M} \n")
        #     break
        
        phi_list.append(phi_m)
        Z_list.append(Zm)
        
        theta = np.dot(Zm,y_cartola) / np.dot(Zm,Zm)
        
        theta_list.append(theta)

        # novo y
        y_chapeu = y_chapeu + theta * Zm 

        # loadings
        p_m = []
        for j in range(x.shape[1]):
            xj = x[:, j]
            term = np.dot(Zm,xj) / np.dot(Zm, Zm)
            p_m.append(term)
            x[:, j] = xj - term * Zm
        load_list.append(np.array(p_m))
    
    Z_train = np.column_stack(Z_list)

    params = {
        "phi": np.vstack(phi_list),     
        "loadings": np.vstack(load_list), 
        "theta": np.array(theta_list),
        "y_mean": y_mean,
    }
    return Z_train, params

# Pré-requisito: x padronizado 
def PLS_transform(X, parametros):
    X = np.asarray(X).copy()
    phi = parametros['phi']
    loadings = parametros['loadings']

    Z_teste = []

    for m in range(len(phi)):
        phi_m = phi[m]
        load_m = loadings[m]
        
        Z = X @ phi_m
        Z_teste.append(Z)

        Z = Z.reshape(-1, 1)   # vira (n, 1)
        load_m = load_m.reshape(1, -1)   # vira (1, p)

        X = X - Z @ load_m

    return np.column_stack(Z_teste)


##########################################################
# rede neural
def rede_neural(n_features, lr = 0.001):
    modelo = keras.Sequential([
        layers.Dense(40, activation = 'relu', input_shape = ((n_features),)),
        

        layers.Dense(16, activation = 'relu'),
        

        layers.Dense(8, activation = 'relu'),
        layers.Dropout(0.2),
        
        
        layers.Dense(1)
    ])

    modelo.compile(
        optimizer = keras.optimizers.Adam(learning_rate = lr),
        loss = 'mse',
        metrics = ['mae']
    )

    return modelo