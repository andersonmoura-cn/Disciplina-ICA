from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from assimetria import SkewAutoTransformer


# salvando imagem
def img_save(file_name):
    # Caminho base: projeto (DISCIPLINA-ICA)
    base = Path.cwd().parents[1]

    # Caminho da pasta de imagens
    folder = base / "images" / "hw3" / "Q2"
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
    arquivo="resultados_modelos.txt"
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
file_path = base / "results" / "resultados_modelos.txt"
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.unlink(missing_ok=True)

# ----------------- KNN com Pipeline -----------------
# print("KNN")

# def melhor_K(X, y, colunas, max_k=20):
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     resultados = []

#     for k in range(1, max_k + 1):
#         pipe = Pipeline([
#             ("skew", SkewAutoTransformer(columns=colunas)),
#             ("scaler", StandardScaler()),
#             ("knn", KNeighborsClassifier(n_neighbors=k)),
#         ])

#         # model_knn = KNeighborsClassifier(n_neighbors=k)
#         # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         scores = cross_val_score(
#             pipe,
#             X,
#             y,
#             cv=cv,
#             scoring="accuracy"
#         )
#         resultados.append((k, scores.mean(), scores.std()))
    
#     K, acuracia_media, acuracia_std = max(resultados, key=lambda x: x[1])
#     print(f"Melhor K = {K}, acurácia média (CV) = {acuracia_media:.4f} ± {acuracia_std:.4f}")
#     return K, acuracia_media, acuracia_std

# K, acuracia_media, acuracia_std  = melhor_K(X_train, y_train, variaveis)

# # treina final no treino inteiro e testa
# pipe_final = Pipeline([
#     ("skew", SkewAutoTransformer(columns=variaveis)),
#     ("scaler", StandardScaler()),
#     ("knn", KNeighborsClassifier(n_neighbors=K)),
# ])

# pipe_final.fit(X_train, y_train)
# y_pred_knn = pipe_final.predict(X_test)
# print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_knn):.4f}")

################### QDA #######################
print("QDA sem preprocessamento adicional")

model_qda = QuadraticDiscriminantAnalysis()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model_qda,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold(QDA): {scores}")
print(f"Acurácia média (CV) - (QDA): {scores.mean():.4f}")
print(f"Desvio padrão (QDA): {scores.std():.4f}")


model_qda.fit(X_train, y_train)
y_pred_qda = model_qda.predict(X_test)

acc_qda = accuracy_score(y_test, y_pred_qda)
cm_qda = confusion_matrix(y_test, y_pred_qda)

print(f"Acurácia no teste(QDA): {acc_qda:.4f}")
print(f"Matrix QDA sem preprocessamento adicional: \n {cm_qda}")

disp_qda = ConfusionMatrixDisplay(confusion_matrix=cm_qda, display_labels=noms_classes)
disp_qda.plot()
img_save("matriz_confusao_qda")

salvar_resultado(
    nome_metodo="QDA",
    preprocessamento="sem_preprocessamento_adicional",
    acuracia=acc_qda,
    matriz_confusao=cm_qda
)
# ----------------------------------------------------------
print("QDA com scale")

pipe_qda = Pipeline([
    ("scaler", StandardScaler()),
    ("qda", QuadraticDiscriminantAnalysis())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_qda,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold(QDA): {scores}")
print(f"Acurácia média (CV) - (QDA): {scores.mean():.4f}")
print(f"Desvio padrão (QDA): {scores.std():.4f}")


pipe_qda.fit(X_train, y_train)
y_pred_qda = pipe_qda.predict(X_test)

acc_qda_scale = accuracy_score(y_test, y_pred_qda)
cm_qda_scale = confusion_matrix(y_test, y_pred_qda)

print(f"Acurácia no teste(QDA): {acc_qda_scale:.4f}")
print(f"Matrix QDA scale: \n {cm_qda_scale}")

disp_qda_scale = ConfusionMatrixDisplay(confusion_matrix=cm_qda_scale, display_labels=noms_classes)
disp_qda_scale.plot()
img_save("matriz_confusao_qda_scale")

salvar_resultado(
    nome_metodo="QDA",
    preprocessamento="scale",
    acuracia=acc_qda_scale,
    matriz_confusao=cm_qda_scale
)

# ----------------------------------------------------------
print("QDA com scale+skew")

pipe_qda = Pipeline([
    ("skew", SkewAutoTransformer(columns=variaveis)),
    ("scaler", StandardScaler()),
    ("qda", QuadraticDiscriminantAnalysis())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_qda,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold(QDA): {scores}")
print(f"Acurácia média (CV) - (QDA): {scores.mean():.4f}")
print(f"Desvio padrão (QDA): {scores.std():.4f}")


pipe_qda.fit(X_train, y_train)
y_pred_qda = pipe_qda.predict(X_test)

acc_qda = accuracy_score(y_test, y_pred_qda)
cm_qda_prep = confusion_matrix(y_test, y_pred_qda)

print(f"Acurácia no teste(QDA): {acc_qda:.4f}")
print(f"Matrix QDA skew+scale: \n {cm_qda_prep}")

disp_qda_prep = ConfusionMatrixDisplay(confusion_matrix=cm_qda_prep, display_labels=noms_classes)
disp_qda_prep.plot()
img_save("matriz_confusao_qda_prep")

salvar_resultado(
    nome_metodo="QDA",
    preprocessamento="skew+scale",
    acuracia=acc_qda,
    matriz_confusao=cm_qda_prep
)


# ----------------- PERCEPTRON -----------------
# print("PERCEPTRON")

# # model_perceptron = Perceptron(max_iter=1000, random_state=42)

# pipe_perceptron = Pipeline([
#     # ("skew", SkewAutoTransformer(columns=variaveis)),  # parece que normalmente não ajuda Perceptron
#     ("scaler", StandardScaler()),
#     ("perceptron", Perceptron(
#         max_iter=2000,        # mais iterações
#         tol=1e-3,             # critério de parada
#         random_state=42
#     ))
# ])


# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(
#     pipe_perceptron,
#     X_train,
#     y_train,
#     cv=cv,
#     scoring="accuracy",
#     n_jobs=-1
# )

# print(f"Acurácia por fold: {scores}")
# print(f"Acurácia média (CV): {scores.mean():.4f}")
# print(f"Desvio padrão: {scores.std():.4f}")

# # treina no conjunto de treino completo
# pipe_perceptron.fit(X_train, y_train)
# y_pred_p = pipe_perceptron.predict(X_test)
# print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_p):.4f}")


######### MLP###########
print("MLP so com scale")

pipe_mlp = Pipeline([
    # ("skew", SkewAutoTransformer(columns=variaveis)), 
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(50, 50),
        solver="adam",
        max_iter=2000,          # mais iterações
        early_stopping=True,    # para sozinho se não melhorar
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=42
    ))
])


# model_mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=300)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_mlp,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold: {scores}")
print(f"Acurácia média (CV): {scores.mean():.4f}")
print(f"Desvio padrão: {scores.std():.4f}")

pipe_mlp.fit(X_train, y_train)
y_pred_mlp = pipe_mlp.predict(X_test)

acc_mlp = accuracy_score(y_test, y_pred_mlp)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

print(f"Acurácia no teste(MLP): {acc_mlp:.4f}")
print(f"Matrix MLP scaler: \n {cm_mlp}")

disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=noms_classes)
disp_mlp.plot()
img_save("matriz_confusao_mlp")

salvar_resultado(
    nome_metodo="MLP",
    preprocessamento="scale",
    acuracia=acc_mlp,
    matriz_confusao=cm_mlp
)

# -------------------------------------------------------------
print("MLP com scale+skew")

pipe_mlp = Pipeline([
    ("skew", SkewAutoTransformer(columns=variaveis)),  # pesa mt
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(50, 50),
        solver="adam",
        max_iter=2000,          # mais iterações
        early_stopping=True,    # para sozinho se não melhorar
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=42
    ))
])


# model_mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=300)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_mlp,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold: {scores}")
print(f"Acurácia média (CV): {scores.mean():.4f}")
print(f"Desvio padrão: {scores.std():.4f}")

pipe_mlp.fit(X_train, y_train)
y_pred_mlp = pipe_mlp.predict(X_test)

acc_mlp = accuracy_score(y_test, y_pred_mlp)
cm_mlp_skew = confusion_matrix(y_test, y_pred_mlp)

print(f"Acurácia no teste(MLP): {acc_mlp:.4f}")
print(f"Matrix MLP scaler+skew: \n {cm_mlp_skew}")

disp_mlp_skew = ConfusionMatrixDisplay(confusion_matrix=cm_mlp_skew, display_labels=noms_classes)
disp_mlp_skew.plot()
img_save("matriz_confusao_mlp_skew")

salvar_resultado(
    nome_metodo="MLP",
    preprocessamento="scale+skew",
    acuracia=acc_mlp,
    matriz_confusao=cm_mlp_skew
)



########### MATRIZ DE CONFUSÃO ###########


# 1. Calcular a Matriz de Confusão
# cm_knn = confusion_matrix(y_test, y_pred_knn)
# cm_qda = confusion_matrix(y_test, y_pred_qda)
# cm_p = confusion_matrix(y_test, y_pred_p)
# cm_mlp = confusion_matrix(y_test, y_pred_mlp)

# print(cm_knn)
# print(cm_qda)
# print(cm_p)
# print(cm_mlp)

# 2. Apresentar a matriz visualmente (para facilitar a leitura).
# noms_classes = ["Com Estresse", "Sem Estresse"]
# disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=noms_classes)
# disp_qda = ConfusionMatrixDisplay(confusion_matrix=cm_qda, display_labels=noms_classes)
# disp_p = ConfusionMatrixDisplay(confusion_matrix=cm_p, display_labels=noms_classes)
# disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=noms_classes)

# disp_knn.plot()
# img_save("matriz_confusao_knn")
# disp_qda.plot()
# img_save("matriz_confusao_qda")
# disp_p.plot()
# img_save("matriz_confusao_p")
# disp_mlp.plot()
# img_save("matriz_confusao_mlp")


