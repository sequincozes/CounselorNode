import numpy as np
import matplotlib.pyplot as plt

# Métricas computadas (macro) a partir das matrizes (ignorando coluna N/A)
# Cenários: 0% = com CN (baseline), 50% = envenenamento médio, 100% = envenenamento total
labels = ["Baseline (0%)", "Médio (50%)", "Total (100%)"]

accuracy = np.array([0.997957, 0.594292, 0.138664])
recall   = np.array([0.997971, 0.543612, 0.093579])
f1       = np.array([0.997653, 0.571871, 0.081185])

x = np.arange(len(labels))
w = 0.25  # largura das barras

plt.figure(figsize=(8, 4.2))

# Barras (sem definir cores, para respeitar seu padrão; o matplotlib escolhe por padrão)
plt.bar(x - w, accuracy, width=w, label="Acurácia")
plt.bar(x,     recall,   width=w, label="Recall", hatch="xxx", edgecolor="black", linewidth=0.8)
plt.bar(x + w, f1,       width=w, label="F1",     hatch="////", edgecolor="black", linewidth=0.8)

# Eixos e grade
plt.ylim(0, 1.0)
plt.ylabel("Score", fontsize=12)
plt.xticks(x, labels, fontsize=11)

plt.grid(axis="y", linestyle=":", linewidth=1)
plt.gca().set_axisbelow(True)

# Legenda no topo (similar à figura)
plt.legend(loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.12))

plt.tight_layout()
plt.show()
