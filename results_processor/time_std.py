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

# ---------------------------------
# 1) GRÁFICO DE LINHAS – evolução
# ---------------------------------
fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(
    df[rodada_col],
    df[local_col],
    label="Decisão local",
    linewidth=1.8
)

ax.plot(
    df[rodada_col],
    df[conf_col],
    label="Resolução com conflito",
    linewidth=1.8
)

ax.set_xlabel("Rodada", fontweight="bold")
ax.set_ylabel("Tempo (ms)", fontweight="bold")

ax.legend(frameon=False)

plt.tight_layout(pad=0.2)

plt.savefig(
    "evolucao_tempo_local_vs_conflito.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()

# ---------------------------------
# 2) GRÁFICO DE BARRAS – média + desvio padrão
# ---------------------------------
media_local = df[local_col].mean(skipna=True)
std_local   = df[local_col].std(skipna=True)

media_conf  = df[conf_col].mean(skipna=True)
std_conf    = df[conf_col].std(skipna=True)

fig, ax = plt.subplots(figsize=(5, 3))

ax.bar(
    ["Decisão local", "Resolução com conflito"],
    [media_local, media_conf],
    yerr=[std_local, std_conf],
    capsize=6
)

ax.set_ylabel("Tempo médio (ms)", fontweight="bold")
ax.set_xlabel("Tipo de decisão", fontweight="bold")

plt.tight_layout(pad=0.2)

plt.savefig(
    "tempo_medio_com_desvio_padrao.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
