import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# Entrada
# ---------------------------------
CSV_PATH = "Acerto 1000 rodadas - Sheet5.csv"

# ---------------------------------
# Leitura
# ---------------------------------
df = pd.read_csv(CSV_PATH)

rodada_col = df.columns[0]
local_col  = df.columns[1]   # Decisões Locais
conf_col   = df.columns[2]   # Conselhos

# Converte para numérico
df[local_col] = pd.to_numeric(df[local_col], errors="coerce")
df[conf_col]  = pd.to_numeric(df[conf_col], errors="coerce")

# -------------
import numpy as np
from scipy import stats

# ---------------------------------
# 2) GRÁFICO DE BARRAS – média + IC 99%
# ---------------------------------

# Remove NaNs explicitamente
local_vals = df[local_col].dropna()
conf_vals  = df[conf_col].dropna()

# Médias
media_local = local_vals.mean()
media_conf  = conf_vals.mean()

# Tamanhos das amostras
n_local = len(local_vals)
n_conf  = len(conf_vals)

# Desvios padrão amostrais
std_local = local_vals.std(ddof=1)
std_conf  = conf_vals.std(ddof=1)

# Valor crítico t para 99%
t_local = stats.t.ppf(0.95, df=n_local - 1)
t_conf  = stats.t.ppf(0.95, df=n_conf - 1)

# Intervalo de confiança (erro)
ic_local = t_local * (std_local / np.sqrt(n_local))
ic_conf  = t_conf  * (std_conf  / np.sqrt(n_conf))

# Plot
fig, ax = plt.subplots(figsize=(5, 3))

ax.bar(
    ["Decisão local", "Resolução com conflito"],
    [media_local, media_conf],
    yerr=[ic_local, ic_conf],
    capsize=6
)

ax.set_ylabel("Tempo médio (ms)", fontweight="bold")
ax.set_xlabel("Tipo de decisão", fontweight="bold")

plt.tight_layout(pad=0.2)

plt.savefig(
    "tempo_medio_com_ic_95.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
