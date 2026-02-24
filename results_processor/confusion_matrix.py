import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Labels
# ---------------------------------
true_labels = ["benign", "FDI", "evil_twin"]
pred_labels = ["benign", "FDI", "evil_twin", "N/A"]

# ---------------------------------
# Matriz de confusão (3 x 4)
# ---------------------------------
cm_clean = np.array([
    [478, 1, 0, 21],
    [0, 249, 1, 0],
    [0, 0, 250, 0]
])

cm_50 = np.array([
    [365, 116, 0, 19],
    [170, 79, 1, 0],
    [111, 0, 139, 0]
])

cm_100 = np.array([
    [137, 351, 0, 12],
    [250, 0, 0, 0],
    [250, 0, 0, 0]
])

cm = cm_100
# ---------------------------------
# Plot
# ---------------------------------
fig, ax = plt.subplots(figsize=(5, 3))

im = ax.imshow(cm, cmap="RdYlGn", alpha=0.75)

# Ticks
ax.set_xticks(np.arange(len(pred_labels)))
ax.set_yticks(np.arange(len(true_labels)))

ax.set_xticklabels(pred_labels)
ax.set_yticklabels(true_labels)

# Labels (em português)
ax.set_xlabel("Decisão Tomada", fontweight="bold")
ax.set_ylabel("Rótulo Correto", fontweight="bold")

# Anotações
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            fontsize=10
        )

# Colorbar
plt.colorbar(im)

plt.tight_layout()

# ---------------------------------
# Salvar em PDF (ANTES do show)
# ---------------------------------
plt.savefig(
    "confusion_matrix_100.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
