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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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

# ----------------- KNN com Pipeline -----------------
print("KNN")

def melhor_K(X, y, colunas, max_k=20):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []

    for k in range(1, max_k + 1):
        pipe = Pipeline([
            ("skew", SkewAutoTransformer(columns=colunas)),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])

        # model_knn = KNeighborsClassifier(n_neighbors=k)
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )
        resultados.append((k, scores.mean(), scores.std()))
    
    K, acuracia_media, acuracia_std = max(resultados, key=lambda x: x[1])
    print(f"Melhor K = {K}, acurácia média (CV) = {acuracia_media:.4f} ± {acuracia_std:.4f}")
    return K, acuracia_media, acuracia_std

K, acuracia_media, acuracia_std  = melhor_K(X_train, y_train, variaveis)

# treina final no treino inteiro e testa
pipe_final = Pipeline([
    ("skew", SkewAutoTransformer(columns=variaveis)),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=K)),
])

pipe_final.fit(X_train, y_train)
y_pred_knn = pipe_final.predict(X_test)
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_knn):.4f}")

################### QDA #######################
print("QDA")

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
print(f"Acurácia no teste(QDA): {accuracy_score(y_test, y_pred_qda):.4f}")

################### NN ##################

########## PERCEPTRON #########
print("PERCEPTRON")

# model_perceptron = Perceptron(max_iter=1000, random_state=42)

pipe_perceptron = Pipeline([
    # ("skew", SkewAutoTransformer(columns=variaveis)),  # parece que normalmente não ajuda Perceptron
    ("scaler", StandardScaler()),
    ("perceptron", Perceptron(
        max_iter=2000,        # mais iterações
        tol=1e-3,             # critério de parada
        random_state=42
    ))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    pipe_perceptron,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"Acurácia por fold: {scores}")
print(f"Acurácia média (CV): {scores.mean():.4f}")
print(f"Desvio padrão: {scores.std():.4f}")

# treina no conjunto de treino completo
pipe_perceptron.fit(X_train, y_train)
y_pred_p = pipe_perceptron.predict(X_test)
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_p):.4f}")


######### MLP ###########
print("MLP")


pipe_mlp = Pipeline([
    # ("skew", SkewAutoTransformer(columns=variaveis)),  # pesa mt e pode não ajudar 
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
print(accuracy_score(y_test, y_pred_mlp))




########### MATRIZ DE CONFUSÃO ###########


# 1. Calcular a Matriz de Confusão
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_qda = confusion_matrix(y_test, y_pred_qda)
cm_p = confusion_matrix(y_test, y_pred_p)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

print(cm_knn)
print(cm_qda)
print(cm_p)
print(cm_mlp)

# 2. Apresentar a matriz visualmente (para facilitar a leitura).
noms_classes = ["Com Estresse", "Sem Estresse"]
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=noms_classes)
disp_qda = ConfusionMatrixDisplay(confusion_matrix=cm_qda, display_labels=noms_classes)
disp_p = ConfusionMatrixDisplay(confusion_matrix=cm_p, display_labels=noms_classes)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=noms_classes)


disp_knn.plot()
img_save("matriz_confusao_knn")
disp_qda.plot()
img_save("matriz_confusao_qda")
disp_p.plot()
img_save("matriz_confusao_p")
disp_mlp.plot()
img_save("matriz_confusao_mlp")