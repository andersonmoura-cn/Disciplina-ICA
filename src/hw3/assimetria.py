import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Transformer sklearn-style que escolhe automatimante para cada coluna a transformação (Original/Log/Box-Cox/Yeo-Johnson/Tangente/Raiz) que
#    minimiza skew no treino (fit) e reaplica a mesma configuração no transform.
class SkewAutoTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, penalty=999.0):
        """
        columns: lista de colunas para transformar.
                 - columns = None e input = DataFrame: usa todas as colunas.
                 - input for numpy: columns deve ser None (transforma todas).
        penalty: valor grande para penalizar métodos inválidos.
        """
        self.columns = columns
        self.penalty = penalty

    def _to_dataframe(self, X):
        # Converte X para DataFrame e retorna (df, colnames).
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            colnames = list(df.columns)
            return df, colnames

        # numpy array
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X deve ser 2D (n_amostras, n_features).")
        colnames = [f"x{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=colnames)
        return df, colnames

    def fit(self, X, y=None):
        df, all_cols = self._to_dataframe(X)

        # define colunas a transformar
        if self.columns is None:
            cols = all_cols
        else:
            # se user passou nomes, valida que existem
            missing = [c for c in self.columns if c not in df.columns]
            if missing:
                raise ValueError(f"Colunas não encontradas em X: {missing}")
            cols = list(self.columns)

        info_transf = {}

        for col in cols:
            data = df[col].astype(float)

            # 1. Skew Original
            original_skew = data.skew()

            # Dicionário temporário para guardar skews desta variável
            metodos = {"Original": abs(original_skew)}
            series_temp = {"Original": data}

            # 2. Logaritmo (só se min >= 0)
            if data.min() >= 0:
                log_data = np.log1p(data)
                metodos["Log"] = abs(pd.Series(log_data, index=data.index).skew())
                series_temp["Log"] = pd.Series(log_data, index=data.index)
            else:
                metodos["Log"] = self.penalty

            # 3. Box-Cox
            shift_bc = None
            lmbda_bc = None
            try:
                # Se tiver 0 ou negativo, deslocamos os dados
                shift_bc = 0.0
                if data.min() <= 0:
                    shift_bc = abs(data.min()) + 1.0
                boxcox_arr, lmbda_bc = stats.boxcox(data + shift_bc)
                boxcox_s = pd.Series(boxcox_arr, index=data.index)
                metodos["Box-Cox"] = abs(boxcox_s.skew())
                series_temp["Box-Cox"] = boxcox_s
            except Exception:
                metodos["Box-Cox"] = self.penalty

            # 4. Yeo-Johnson (funciona com negativos)
            lmbda_yj = None
            try:
                yeo_arr, lmbda_yj = stats.yeojohnson(data)
                yeo_s = pd.Series(yeo_arr, index=data.index)
                metodos["Yeo-Johnson"] = abs(yeo_s.skew())
                series_temp["Yeo-Johnson"] = yeo_s
            except Exception:
                metodos["Yeo-Johnson"] = self.penalty

            # 5. Tangente
            try:
                tan_arr = np.tan(data)
                tan_s = pd.Series(tan_arr, index=data.index)
                # Tangente pode gerar valores absurdos, então verifica se o skew não é NaN/Inf
                if np.isfinite(tan_s).all():
                    sk_tan = tan_s.skew()
                    if np.isfinite(sk_tan):
                        metodos["Tangente"] = abs(sk_tan)
                        series_temp["Tangente"] = tan_s
                    else:
                        metodos["Tangente"] = self.penalty
                else:
                    metodos["Tangente"] = self.penalty
            except Exception:
                metodos["Tangente"] = self.penalty

            # 6. Raiz Quadrada
            shift_sqrt = None
            try:
                shift_sqrt = 0.0
                min_val = data.min()
                # Se houver negativos, shift para a raiz
                if min_val < 0:
                    shift_sqrt = abs(min_val)
                raiz_arr = np.sqrt(data + shift_sqrt)
                raiz_s = pd.Series(raiz_arr, index=data.index)
                sk_raiz = abs(raiz_s.skew())
                if np.isfinite(sk_raiz):
                    metodos["Raiz"] = sk_raiz
                    series_temp["Raiz"] = raiz_s
                else:
                    metodos["Raiz"] = self.penalty
            except Exception:
                metodos["Raiz"] = self.penalty

            # escolhe o melhor
            melhor_metodo = min(metodos, key=metodos.get)

            # salva config
            cfg = {"method": melhor_metodo}
            if melhor_metodo == "Box-Cox":
                cfg["shift"] = shift_bc
                cfg["lambda"] = lmbda_bc
            elif melhor_metodo == "Yeo-Johnson":
                cfg["lambda"] = lmbda_yj
            elif melhor_metodo == "Raiz":
                cfg["shift"] = shift_sqrt

            info_transf[col] = cfg

        self.info_ = info_transf
        self.columns_ = cols  # colunas realmente transformadas
        self.input_columns_ = all_cols  # colunas da entrada
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["info_", "columns_", "input_columns_"])

        df, all_cols = self._to_dataframe(X)

        # garante mesma estrutura básica
        # (se entrar DataFrame com colunas diferentes, falha)
        if set(all_cols) != set(self.input_columns_):
            # se for DataFrame, a ordem e nomes importam
            raise ValueError(
                "As colunas de X no transform não batem com as do fit.\n"
                f"fit: {self.input_columns_}\n"
                f"transform: {all_cols}"
            )

        out = df.copy()

        for col in self.columns_:
            cfg = self.info_[col]
            metodo = cfg["method"]
            data = out[col].astype(float)

            if metodo == "Original":
                out[col] = data

            elif metodo == "Raiz":
                shift = cfg.get("shift", 0.0) or 0.0
                out[col] = np.sqrt(data + shift)

            elif metodo == "Log":
                # No fit, só escolhe Log, se min>=0
                out[col] = np.log1p(data)

            elif metodo == "Box-Cox":
                shift = cfg["shift"]
                lmbda = cfg["lambda"]
                # stats.boxcox retorna numpy array; preservando índice
                out[col] = pd.Series(stats.boxcox(data + shift, lmbda=lmbda), index=out.index)

            elif metodo == "Yeo-Johnson":
                lmbda = cfg["lambda"]
                out[col] = pd.Series(stats.yeojohnson(data, lmbda=lmbda), index=out.index)

            elif metodo == "Tangente":
                out[col] = np.tan(data)

            else:
                raise ValueError(f"Método desconhecido: {metodo}")

        # se passou numpy na entrada, devolve numpy (pra combinar com sklearn)
        if isinstance(X, pd.DataFrame):
            return out
        return out.to_numpy()
