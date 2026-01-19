from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from assimetria import SkewAutoTransformer


# salvando imagem
def img_save(file_name):
    # Caminho base: projeto (DISCIPLINA-ICA)
    base = Path.cwd().parents[1]

    # Caminho da pasta de imagens
    folder = base / "images" / "hw3" / "Q1"
    folder.mkdir(parents=True, exist_ok=True)

    # Caminho completo do arquivo
    file_path = folder / f"{file_name}.png"

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Imagem salva em: {file_path}")


# ------------------Salvar resultados-----
def salvar_resultado(
    nome_metodo,
    preprocessamento,
    acuracia,
    matriz_confusao,
    arquivo="resultados_modelos_Q1.txt"
):
    base = Path.cwd().parents[1]
    caminho = base / "results"
    caminho.mkdir(parents=True, exist_ok=True)

    file_path = caminho / arquivo

    with open(file_path, mode="a", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Método: {nome_metodo}\n")
        f.write(f"Pré-processamento: {preprocessamento}\n")
        f.write(f"Acurácia no teste: {acuracia:.4f}\n")
        f.write("Matriz de confusão:\n")

        for linha in matriz_confusao:
            f.write(f"{linha.tolist()}\n")

        f.write("=" * 50 + "\n\n")

    print(f"Resultado salvo em: {file_path}")


# ----------------- dados -----------------
base = Path(__file__).resolve().parent.parent.parent
arquivo_treino = base / "data/hw3" / "df_treino.csv"
df_treino = pd.read_csv(arquivo_treino)
arquivo_teste = base / "data/hw3" / "df_teste.csv"
df_teste = pd.read_csv(arquivo_teste)

X_train = df_treino.drop("condition", axis=1)
y_train = df_treino["condition"]

X_test = df_teste.drop("condition", axis=1)
y_test = df_teste["condition"]

variaveis = list(X_train.columns)
noms_classes = ["Com Estresse", "Sem Estresse"]


# --------- limpar arquivo de resultados uma vez por execução ----------
file_path = base / "results" / "resultados_modelos_Q1.txt"
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.unlink(missing_ok=True)


# ----------------- LDA -----------------

# Inicialização
# modelo_lda = LinearDiscriminantAnalysis()
print("LDA sem processamento adicional")


lda = LinearDiscriminantAnalysis()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    lda,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold (LDA): {scores}")
print(f"Acurácia média (CV) - (LDA): {scores.mean():.4f}")
print(f"Desvio padrão (LDA): {scores.std():.4f}")

lda.fit(X_train, y_train)
predicao = lda.predict(X_test)

acc_lda = accuracy_score(y_test, predicao)
cm_lda = confusion_matrix(y_test, predicao)

print(f"Acurácia (teste) - LDA sem preprocessamento adicional: {acc_lda:.4f}")
print(f"Matrix LDA sem preprocessamento adicional: \n {cm_lda}")

disp_lda = ConfusionMatrixDisplay(confusion_matrix=cm_lda, display_labels=noms_classes)
disp_lda.plot()
img_save("matriz_confusao_lda")

salvar_resultado(
    nome_metodo="LDA",
    preprocessamento="sem_preprocessamento_adicional",
    acuracia=acc_lda,
    matriz_confusao=cm_lda
)

# --------------------------------------------------

print("LDA + scaler")

pipe_lda = Pipeline([
    # ("skew", SkewAutoTransformer(columns=variaveis)),
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_lda,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold (LDA): {scores}")
print(f"Acurácia média (CV) - (LDA): {scores.mean():.4f}")
print(f"Desvio padrão (LDA): {scores.std():.4f}")

# Treino
pipe_lda.fit(X_train, y_train)
predicao = pipe_lda.predict(X_test)
# modelo_lda.fit(X_train, y_train)

# Predição
# predicao = modelo_lda.predict(X_test)

acc_lda_scale = accuracy_score(y_test, predicao)
cm_lda_scale = confusion_matrix(y_test, predicao)

print(f"Acurácia (teste) - LDA scale: {acc_lda_scale:.4f}")
print(f"Matrix LDA scale: \n {cm_lda_scale}")

disp_lda_scale = ConfusionMatrixDisplay(confusion_matrix=cm_lda_scale, display_labels=noms_classes)
disp_lda_scale.plot()
img_save("matriz_confusao_lda_scale")

salvar_resultado(
    nome_metodo="LDA",
    preprocessamento="scale",
    acuracia=acc_lda_scale,
    matriz_confusao=cm_lda_scale
)


# --------------------------------------------------

print("LDA + skew+scaler")

pipe_lda = Pipeline([
    ("skew", SkewAutoTransformer(columns=variaveis)),
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_lda,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold (LDA): {scores}")
print(f"Acurácia média (CV) - (LDA): {scores.mean():.4f}")
print(f"Desvio padrão (LDA): {scores.std():.4f}")

# Treino
pipe_lda.fit(X_train, y_train)
predicao = pipe_lda.predict(X_test)
# modelo_lda.fit(X_train, y_train)

# Predição
# predicao = modelo_lda.predict(X_test)

acc_lda_skew = accuracy_score(y_test, predicao)
cm_lda_skew = confusion_matrix(y_test, predicao)

print(f"Acurácia (teste) - LDA skew+scale: {acc_lda_skew:.4f}")
print(f"Matrix LDA skew+scale: \n {cm_lda_skew}")

disp_lda_skew = ConfusionMatrixDisplay(confusion_matrix=cm_lda_skew, display_labels=noms_classes)
disp_lda_skew.plot()
img_save("matriz_confusao_lda_skew")

salvar_resultado(
    nome_metodo="LDA",
    preprocessamento="skew+scale",
    acuracia=acc_lda_skew,
    matriz_confusao=cm_lda_skew
)


########### MATRIZ DE CONFUSÃO ###########


# Matriz de Confusão
# cm = confusion_matrix(y_test, predicao)
# print(cm)

# # Apresentar a matriz visualmente
# noms_classes = ["Com Estresse", "Sem Estresse"]
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=noms_classes)


# disp.plot()
# print(f"Acurácia(LDA): {accuracy_score(y_test,predicao)}")
# img_save("matriz_confusao")