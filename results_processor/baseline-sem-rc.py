import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ======================================================
# CONFIGURAÇÃO FIXA (PADRÃO SBRC / ZERO-DAY)
# ======================================================
TRAIN_CSV = "../datasets/sbrc/treino_no1.csv"
TEST_CSV  = "../datasets/sbrc/zero_day.csv"
TARGET_COL = None  # última coluna

SAVE_PDF = True
OUT_DIR = "figures"

UNKNOWN_PRED_LABEL = "N/A"

# Ordem EXATA das classes (linhas = ground truth)
ALL_TRUE_LABELS = ["benign", "FDI", "evil_twin"]

CLASSIFIERS = {
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "SVC": SVC(random_state=42),
}


# ======================================================
# IO
# ======================================================
def load_csv(path, target_col=None):
    df = pd.read_csv(path)

    if target_col is None:
        target_col = df.columns[-1]

    y = df[target_col].astype(str).values
    X = (
        df.drop(columns=[target_col])
          .apply(pd.to_numeric, errors="coerce")
          .fillna(0.0)
          .values
    )
    return X, y


# ======================================================
# MATRIZ RETANGULAR (ZERO-DAY CORRETA)
# ======================================================
def rectangular_confusion_matrix(y_true, y_pred, row_labels, col_labels, unknown_col):
    row_index = {lab: i for i, lab in enumerate(row_labels)}
    col_index = {lab: j for j, lab in enumerate(col_labels)}

    cm = np.zeros((len(row_labels), len(col_labels)), dtype=float)

    for yt, yp in zip(y_true, y_pred):
        if yt not in row_index:
            continue
        if yp not in col_index:
            yp = unknown_col
        cm[row_index[yt], col_index[yp]] += 1.0

    return cm


# ======================================================
# PLOT (ESTILO FIXO)
# ======================================================
def plot_heatmap(cm, x_labels, y_labels, savepath=None):
    fig, ax = plt.subplots(figsize=(5, 2.9))

    im = ax.imshow(cm, cmap="RdYlGn", alpha=0.75)
    ax.set_aspect("auto")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    # Classes SEM negrito
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Apenas rótulos externos em negrito
    ax.set_xlabel("Decisão Tomada", fontweight="bold")
    ax.set_ylabel("Rótulo Correto", fontweight="bold")

    # Valores nas células
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10)

    plt.colorbar(im)
    plt.tight_layout(pad=0.1)

    if savepath:
        plt.savefig(savepath, format="pdf", bbox_inches="tight")

    plt.show()


# ======================================================
# MAIN
# ======================================================
if SAVE_PDF:
    os.makedirs(OUT_DIR, exist_ok=True)

X_train_raw, y_train = load_csv(TRAIN_CSV, TARGET_COL)
X_test_raw,  y_test  = load_csv(TEST_CSV,  TARGET_COL)

# Classes conhecidas = só as do TREINO
known_labels = sorted(list(set(y_train)))
col_labels = known_labels + [UNKNOWN_PRED_LABEL]

# Escalonamento correto (fit só no treino)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Identificação de zero-day (ground truth fora do treino)
known_set = set(known_labels)
zeroday_labels = sorted({lab for lab in y_test if lab not in known_set})

print("Zero-day no TESTE:", zeroday_labels)

cms = {}

for name, clf in CLASSIFIERS.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test).astype(str)

    cm = rectangular_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        row_labels=ALL_TRUE_LABELS,
        col_labels=col_labels,
        unknown_col=UNKNOWN_PRED_LABEL
    )

    cms[name] = cm

    plot_heatmap(
        cm,
        x_labels=col_labels,
        y_labels=ALL_TRUE_LABELS,
        savepath=f"{OUT_DIR}/{name}_cm_zero_day.pdf"
    )

    # Replay zero-day (interpretação científica)
    if zeroday_labels:
        print(f"[{name}] Replay zero-day:")
        for z in zeroday_labels:
            idx = np.where(y_test == z)[0]
            preds = y_pred[idx]
            vals, cnts = np.unique(preds, return_counts=True)
            replay = ", ".join([f"{v}={c}" for v, c in zip(vals, cnts)])
            print(f"  {z}: {replay}")
        print()

# Matriz média
avg_cm = np.mean(np.stack(list(cms.values())), axis=0)

plot_heatmap(
    avg_cm,
    x_labels=col_labels,
    y_labels=ALL_TRUE_LABELS,
    savepath=f"{OUT_DIR}/AVERAGE_cm_zero_day.pdf"
)
