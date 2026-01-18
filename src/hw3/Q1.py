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

# ----------------- LDA -----------------

# Inicialização
# modelo_lda = LinearDiscriminantAnalysis()
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

print(f"Acurácia (teste) - (LDA): {accuracy_score(y_test, predicao):.4f}")


########### MATRIZ DE CONFUSÃO ###########


# Matriz de Confusão
cm = confusion_matrix(y_test, predicao)
print(cm)

# Apresentar a matriz visualmente
noms_classes = ["Com Estresse", "Sem Estresse"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=noms_classes)


disp.plot()
print(f"Acurácia(LDA): {accuracy_score(y_test,predicao)}")
img_save("matriz_confusao")