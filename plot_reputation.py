"""
plot_reputation.py
==================
Gera gráficos da evolução da reputação atribuída entre os nós da rede CounselorNode
ao longo dos desafios emitidos pelo mecanismo de defesa (ReputationManager).

O script gera:
  1. Painel consolidado com 3 gráficos (um para cada nó avaliador), onde cada gráfico
     apresenta duas curvas representando a reputação atribuída aos outros dois nós.
  2. Gráficos individuais de alta resolução para cada nó avaliador.
  3. Exportação em múltiplos formatos:
     - PNG (300 DPI para relatórios/artigos)
     - PDF (vetorial para LaTeX/Overleaf na dissertação)
     - SVG (vetorial nativo, abre diretamente no navegador mesmo sem dependências)
     - HTML interativo com os 3 gráficos para visualização imediata

Uso:
----
  # Gera gráficos com configurações padrão (salva em logs/plots):
  python plot_reputation.py

  # Especifica diretório customizado de logs e saída:
  python plot_reputation.py --log-dir logs --output-dir logs/plots

  # Gera gráficos em PDF vetorial para dissertação/LaTeX:
  python plot_reputation.py --format pdf

  # Layout vertical (3 linhas x 1 coluna, ideal para folha A4):
  python plot_reputation.py --layout vertical

  # Exibe a janela gráfica interativa (caso ambiente com display):
  python plot_reputation.py --show
"""

import os
import csv
import json
import glob
import argparse
from collections import defaultdict


# =============================================================================
# Configurações de Identidade Visual e Estilo
# =============================================================================
NODE_COLORS = {
    "no_1": "#1f77b4",  # Azul sólido
    "no_2": "#d62728",  # Vermelho (destaque de nó envenenado/isolado)
    "no_3": "#2ca02c",  # Verde esmeralda
    "default_1": "#ff7f0e",
    "default_2": "#9467bd",
}

NODE_DISPLAY_NAMES = {
    "no_1": "Nó 1",
    "no_2": "Nó 2",
    "no_3": "Nó 3",
}

NODE_LINESTYLES = {
    "no_1": "-",
    "no_2": "-",
    "no_3": "-",
}

NODE_MARKERS = {
    "no_1": "o",
    "no_2": "s",
    "no_3": "^",
}


def get_node_label(node_name):
    """Retorna nome formatado do nó para rótulos e títulos."""
    return NODE_DISPLAY_NAMES.get(node_name.lower(), node_name.upper())


def get_node_color(node_name, fallback_idx=0):
    """Retorna cor consistente do nó em qualquer gráfico."""
    key = node_name.lower()
    if key in NODE_COLORS:
        return NODE_COLORS[key]
    fallback_keys = ["default_1", "default_2"]
    return NODE_COLORS[fallback_keys[fallback_idx % len(fallback_keys)]]


# =============================================================================
# Carregamento e Processamento de Dados dos Logs CSV
# =============================================================================
def load_reputation_series(filepath, x_mode="per-peer"):
    """
    Carrega o arquivo CSV de reputação de um nó avaliador e agrupa as séries
    temporais de reputação atribuída a cada par consultado.

    Parâmetros:
    -----------
    filepath : str
        Caminho para o arquivo CSV (ex: logs/no_1_reputacao.csv).
    x_mode : str
        'per-peer' -> Eixo X é o número de desafios respondidos pelo par (1, 2, 3...)
        'timeline' -> Eixo X é o evento global de desafio do nó avaliador (seq_event)

    Retorna:
    --------
    dict: { peer_name: { 'x': [...], 'y': [...], 'estados': [...], 'tau': float, 'last_state': str } }
    """
    series_by_peer = defaultdict(lambda: {
        "x": [],
        "y": [],
        "estados": [],
        "tau": 0.7,
        "last_state": "EVALUATING"
    })

    peer_counter = defaultdict(int)

    if not os.path.exists(filepath):
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            peer = row.get("peer_name", "").strip().lower()
            if not peer:
                continue

            try:
                rep_val = float(row.get("reputacao", 0.0))
            except ValueError:
                continue

            try:
                tau_val = float(row.get("limiar_tau", 0.7))
            except ValueError:
                tau_val = 0.7

            estado = row.get("estado", "EVALUATING").strip()

            if x_mode == "per-peer":
                peer_counter[peer] += 1
                x_val = peer_counter[peer]
            else:
                try:
                    x_val = int(row.get("seq_event", len(series_by_peer[peer]["x"]) + 1))
                except ValueError:
                    x_val = len(series_by_peer[peer]["x"]) + 1

            series_by_peer[peer]["x"].append(x_val)
            series_by_peer[peer]["y"].append(rep_val)
            series_by_peer[peer]["estados"].append(estado)
            series_by_peer[peer]["tau"] = tau_val
            series_by_peer[peer]["last_state"] = estado

    return dict(series_by_peer)


# =============================================================================
# Renderização via Matplotlib (PNG / PDF / JPG / Janela Interativa)
# =============================================================================
def plot_single_node_axis(ax, evaluator_node, peer_series, tau_threshold=0.7, x_mode="per-peer"):
    """
    Plota as duas curvas de reputação em um objeto Axes do matplotlib.
    """
    evaluator_label = get_node_label(evaluator_node)
    ax.set_title(f"Visão do {evaluator_label}", fontsize=13, fontweight="bold", pad=10)

    x_label = "Número de Desafios Respondidos pelo Par" if x_mode == "per-peer" else "Sequência Global de Desafios"
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Reputação", fontsize=11, fontweight="bold")

    # Linha horizontal do limiar Tau
    ax.axhline(
        y=tau_threshold,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        label=f"Limiar de Isolamento (τ = {tau_threshold:.2f})"
    )

    # Zona de isolamento sombreada (abaixo de tau)
    ax.axhspan(
        0.0,
        tau_threshold,
        facecolor="#fee8e7",
        alpha=0.35,
        edgecolor="none"
    )

    max_x = 10

    # Plota as curvas dos pares avaliados
    for idx, (peer_name, data) in enumerate(sorted(peer_series.items())):
        x_vals = data["x"]
        y_vals = data["y"]
        if not x_vals:
            continue

        max_x = max(max_x, max(x_vals))
        peer_label = get_node_label(peer_name)
        color = get_node_color(peer_name, idx)
        marker = NODE_MARKERS.get(peer_name.lower(), "o")
        linestyle = NODE_LINESTYLES.get(peer_name.lower(), "-")

        final_rep = y_vals[-1]
        final_state = data.get("last_state", "N/A")

        # Rótulo informativo na legenda
        legend_label = f"Reputação do {peer_label} (final: {final_rep:.2f} | {final_state})"

        # Curva com marcadores esparsos se houver muitos pontos
        markevery = max(1, len(x_vals) // 15) if len(x_vals) > 20 else 1

        ax.plot(
            x_vals,
            y_vals,
            label=legend_label,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker=marker,
            markersize=5,
            markevery=markevery,
            alpha=0.9
        )

        # Destaca o último ponto alcançado
        ax.plot(
            x_vals[-1],
            y_vals[-1],
            marker="*",
            markersize=10,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=5
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(max_x * 1.05, 5))
    ax.grid(True, linestyle=":", alpha=0.6, color="#b0b0b0")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9, edgecolor="#cccccc")


def generate_matplotlib_plots(node_data, output_dir, tau=0.7,
                              x_mode="per-peer", layout="horizontal",
                              img_format="png", dpi=300, show=False):
    """
    Gera o painel com 3 gráficos e as figuras individuais usando Matplotlib.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected_nodes = ["no_1", "no_2", "no_3"]

    # 1. Gera os 3 gráficos individuais
    for node_name, series in node_data.items():
        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
        plot_single_node_axis(ax, node_name, series, tau_threshold=tau, x_mode=x_mode)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"reputacao_visao_{node_name}.{img_format}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[Matplotlib] Gráfico individual salvo: {out_path}")
        plt.close(fig)

    # 2. Gera a figura consolidada com os 3 gráficos
    nodes_to_plot = [n for n in expected_nodes if n in node_data]
    n_plots = len(nodes_to_plot)

    if n_plots > 0:
        if layout == "horizontal":
            # 1 linha x 3 colunas
            fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 5.2), dpi=dpi, sharey=True)
            if n_plots == 1:
                axes = [axes]
        else:
            # 3 linhas x 1 coluna (vertical, ideal para folhas A4 em relatórios)
            fig, axes = plt.subplots(n_plots, 1, figsize=(9.0, 4.2 * n_plots), dpi=dpi, sharex=False)
            if n_plots == 1:
                axes = [axes]

        fig.suptitle(
            "Evolução da Reputação dos Pares por Nó Avaliador (Mecanismo de Defesa P2P)",
            fontsize=15,
            fontweight="bold",
            y=0.99 if layout == "horizontal" else 0.995
        )

        for ax, node_name in zip(axes, nodes_to_plot):
            series = node_data[node_name]
            plot_single_node_axis(ax, node_name, series, tau_threshold=tau, x_mode=x_mode)

        plt.tight_layout()
        if layout == "horizontal":
            plt.subplots_adjust(top=0.88)
        else:
            plt.subplots_adjust(top=0.93)

        combined_name = f"reputacao_painel_3_nos_{layout}.{img_format}"
        combined_path = os.path.join(output_dir, combined_name)
        plt.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        print(f"[Matplotlib] Painel consolidado salvo: {combined_path}")

        if show:
            plt.show()

        plt.close(fig)


# =============================================================================
# Renderizador Vetorial SVG Nativo (Independente, roda em Python puro)
# =============================================================================
def generate_standalone_svg(evaluator_node, peer_series, tau_threshold=0.7, width=650, height=400):
    """
    Gera um gráfico SVG vetorial puro para um nó avaliador específico,
    com as duas curvas dos nós avaliados, eixos, grade e linha do limiar tau.
    """
    margin = {"top": 45, "right": 30, "bottom": 55, "left": 65}
    pw = width - margin["left"] - margin["right"]
    ph = height - margin["top"] - margin["bottom"]

    m_top = margin["top"]
    m_bottom = margin["bottom"]
    m_left = margin["left"]
    m_right = margin["right"]

    max_x = max([max(d["x"]) for d in peer_series.values() if d["x"]] or [10])
    max_x = max(max_x, 5)

    def to_x(val):
        return m_left + (val / max_x) * pw

    def to_y(val):
        return m_top + (1.0 - val) * ph

    iso_y = to_y(tau_threshold)
    iso_h = ph - (iso_y - m_top)

    svg_lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family: sans-serif;">',
        f'  <!-- Área de Isolamento (abaixo de tau) -->',
        f'  <rect x="{to_x(0):.1f}" y="{iso_y:.1f}" width="{pw:.1f}" height="{iso_h:.1f}" fill="#fee8e7" opacity="0.45" />',
        f'  <!-- Grade horizontal -->'
    ]

    # Grid Y
    for y_step in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y_px = to_y(y_step)
        svg_lines.append(f'  <line x1="{m_left}" y1="{y_px:.1f}" x2="{width - m_right}" y2="{y_px:.1f}" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="3,3" />')
        svg_lines.append(f'  <text x="{m_left - 8}" y="{y_px + 4:.1f}" font-size="11" fill="#555555" text-anchor="end">{y_step:.1f}</text>')

    # Linha do limiar Tau
    y_tau = to_y(tau_threshold)
    svg_lines.append(f'  <line x1="{m_left}" y1="{y_tau:.1f}" x2="{width - m_right}" y2="{y_tau:.1f}" stroke="#7f7f7f" stroke-width="1.8" stroke-dasharray="6,4" />')
    svg_lines.append(f'  <text x="{width - m_right - 5}" y="{y_tau - 6:.1f}" font-size="10" fill="#666666" text-anchor="end" font-weight="bold">Limiar τ = {tau_threshold:.2f}</text>')

    # Eixos X e Y
    svg_lines.append(f'  <line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{height - m_bottom}" stroke="#333333" stroke-width="1.5" />')
    svg_lines.append(f'  <line x1="{m_left}" y1="{height - m_bottom}" x2="{width - m_right}" y2="{height - m_bottom}" stroke="#333333" stroke-width="1.5" />')

    # Rótulos dos eixos
    evaluator_label = get_node_label(evaluator_node)
    svg_lines.append(f'  <text x="{width / 2}" y="26" font-size="14" font-weight="bold" fill="#222222" text-anchor="middle">Visão do {evaluator_label}</text>')
    svg_lines.append(f'  <text x="{width / 2}" y="{height - 12}" font-size="12" font-weight="bold" fill="#333333" text-anchor="middle">Número de Desafios do Par</text>')
    svg_lines.append(f'  <text x="18" y="{height / 2}" font-size="12" font-weight="bold" fill="#333333" text-anchor="middle" transform="rotate(-90, 18, {height / 2})">Reputação</text>')

    # Curvas dos dois nós avaliados
    legend_items = []
    for idx, (peer_name, data) in enumerate(sorted(peer_series.items())):
        x_vals = data["x"]
        y_vals = data["y"]
        if not x_vals:
            continue

        color = get_node_color(peer_name, idx)
        peer_label = get_node_label(peer_name)
        final_rep = y_vals[-1]
        final_state = data.get("last_state", "")

        # Monta caminho SVG (path)
        points = [f"{to_x(x):.1f},{to_y(y):.1f}" for x, y in zip(x_vals, y_vals)]
        svg_lines.append(f'  <polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{" ".join(points)}" opacity="0.9" />')

        # Marcadores pontuais (se houver muitos, amostra)
        step = max(1, len(x_vals) // 15)
        for i in range(0, len(x_vals), step):
            svg_lines.append(f'  <circle cx="{to_x(x_vals[i]):.1f}" cy="{to_y(y_vals[i]):.1f}" r="3.5" fill="{color}" />')

        # Último ponto destacado
        last_x_px = to_x(x_vals[-1])
        last_y_px = to_y(y_vals[-1])
        svg_lines.append(f'  <circle cx="{last_x_px:.1f}" cy="{last_y_px:.1f}" r="5.5" fill="{color}" stroke="#000000" stroke-width="1.2" />')

        legend_items.append((peer_label, color, final_rep, final_state))

    # Caixa de Legenda
    leg_x = width - m_right - 220
    leg_y = height - m_bottom - 65
    svg_lines.append(f'  <rect x="{leg_x}" y="{leg_y}" width="215" height="55" rx="4" fill="#ffffff" stroke="#cccccc" stroke-width="1" opacity="0.92" />')

    for i, (p_lbl, p_col, p_rep, p_st) in enumerate(legend_items):
        item_y = leg_y + 18 + (i * 22)
        svg_lines.append(f'  <line x1="{leg_x + 12}" y1="{item_y}" x2="{leg_x + 32}" y2="{item_y}" stroke="{p_col}" stroke-width="2.5" />')
        svg_lines.append(f'  <circle cx="{leg_x + 22}" cy="{item_y}" r="3.5" fill="{p_col}" />')
        svg_lines.append(f'  <text x="{leg_x + 38}" y="{item_y + 4}" font-size="11" fill="#333333">{p_lbl} (rep: {p_rep:.2f} | {p_st})</text>')

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


def generate_native_svg_and_html(node_data, output_dir, tau=0.7):
    """
    Gera arquivos SVG para cada nó e uma página HTML interativa contendo
    os 3 gráficos dispostos horizontalmente em um painel consolidado.
    Funciona em qualquer computador sem precisar de matplotlib instalado!
    """
    os.makedirs(output_dir, exist_ok=True)
    expected_nodes = ["no_1", "no_2", "no_3"]

    svg_contents = {}

    for node_name in expected_nodes:
        if node_name in node_data:
            series = node_data[node_name]
            svg_str = generate_standalone_svg(node_name, series, tau_threshold=tau, width=620, height=390)
            svg_path = os.path.join(output_dir, f"reputacao_visao_{node_name}.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg_str)
            svg_contents[node_name] = svg_str
            print(f"[SVG Nativo] Salvo: {svg_path}")

    # Gera HTML consolidado com os 3 gráficos
    html_cards = []
    for node_name in expected_nodes:
        if node_name in svg_contents:
            lbl = get_node_label(node_name)
            html_cards.append(f"""
            <div class="chart-card">
                <h3>Visão do {lbl}</h3>
                <div class="svg-container">
                    {svg_contents[node_name]}
                </div>
            </div>
            """)

    html_page = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Reputação dos Nós — Mecanismo de Defesa P2P</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f4f6f9;
            color: #222;
            margin: 0;
            padding: 24px;
        }}
        header {{
            text-align: center;
            margin-bottom: 24px;
        }}
        h1 {{
            font-size: 22px;
            margin: 0 0 8px 0;
            color: #1a252f;
        }}
        p.subtitle {{
            color: #666;
            font-size: 14px;
            margin: 0;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
            gap: 20px;
            max-width: 1900px;
            margin: 0 auto;
        }}
        .chart-card {{
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 16px;
            border: 1px solid #e1e4e8;
        }}
        .chart-card h3 {{
            margin: 0 0 12px 0;
            font-size: 16px;
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        .svg-container {{
            display: flex;
            justify-content: center;
            overflow-x: auto;
        }}
        .badge-legend {{
            margin-top: 20px;
            padding: 12px;
            background: #ffffff;
            border-radius: 6px;
            border: 1px solid #ddd;
            text-align: center;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Evolução da Reputação dos Nós da Rede CounselorNode</h1>
        <p class="subtitle">3 Gráficos: Cada nó avaliador monitorando seus pares com o limiar de isolamento (τ = {tau:.2f})</p>
    </header>

    <div class="grid-container">
        {"".join(html_cards)}
    </div>

    <div class="badge-legend">
        <strong>Identidade Visual:</strong>
        <span style="color: #1f77b4; font-weight: bold; margin: 0 12px;">● Nó 1 (Azul)</span>
        <span style="color: #d62728; font-weight: bold; margin: 0 12px;">● Nó 2 (Vermelho / Envenenado)</span>
        <span style="color: #2ca02c; font-weight: bold; margin: 0 12px;">● Nó 3 (Verde)</span>
        <span style="color: #7f7f7f; font-weight: bold; margin: 0 12px;">--- Limiar de Isolamento (τ = {tau:.2f})</span>
    </div>
</body>
</html>
"""
    html_path = os.path.join(output_dir, "painel_reputacao_3_nos.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_page)
    print(f"[HTML Interativo] Painel visualizável no navegador salvo: {html_path}")


# =============================================================================
# Função Principal (CLI)
# =============================================================================
def main():
    default_tau = 0.7
    config_file = os.path.join(os.path.dirname(__file__), "config", "peer_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            default_tau = float(cfg.get("ml_config", {}).get("defense_tau", 0.7))
        except Exception:
            default_tau = 0.7

    parser = argparse.ArgumentParser(
        description="Plota 3 gráficos da reputação atribuída entre os nós conforme o número de desafios."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Diretório onde estão os arquivos *_reputacao.csv (padrão: logs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/plots",
        help="Diretório onde as figuras geradas serão salvas (padrão: logs/plots)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=default_tau,
        help=f"Limiar de reputação para corte/isolamento (padrão: {default_tau:.2f})"
    )
    parser.add_argument(
        "--x-mode",
        type=str,
        default="per-peer",
        choices=["per-peer", "timeline"],
        help="Modo do eixo X: 'per-peer' (número de desafios do par) ou 'timeline' (seq_event global). Padrão: per-peer"
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="Layout do painel consolidado: 'horizontal' (1x3) ou 'vertical' (3x1). Padrão: horizontal"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "jpg"],
        help="Formato do arquivo de imagem salvo pelo matplotlib (padrão: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolução da imagem em DPI para publicação acadêmica (padrão: 300)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre janela gráfica interativa com os gráficos além de salvar"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  GERADOR DE GRÁFICOS DE REPUTAÇÃO — REDE DE CONSELHOS (DEFESA P2P)")
    print("=" * 80)
    print(f"Diretório de logs : {args.log_dir}")
    print(f"Diretório de saída: {args.output_dir}")
    print(f"Limiar Tau (τ)    : {args.tau}")
    print(f"Modo do Eixo X    : {args.x_mode}")
    print(f"Layout do Painel  : {args.layout}")
    print()

    expected_nodes = ["no_1", "no_2", "no_3"]
    node_data = {}

    for node_name in expected_nodes:
        filepath = os.path.join(args.log_dir, f"{node_name}_reputacao.csv")
        if os.path.exists(filepath):
            series = load_reputation_series(filepath, x_mode=args.x_mode)
            if series:
                node_data[node_name] = series
                n_peers = len(series)
                tot_desafios = sum(len(d["x"]) for d in series.values())
                print(f"[OK] {node_name}: {n_peers} pares avaliados ({tot_desafios} registros no total)")
                for p_name, p_data in series.items():
                    print(f"     -> Avaliou {p_name}: {len(p_data['x'])} desafios | Última rep: {p_data['y'][-1]:.4f} ({p_data['last_state']})")
            else:
                print(f"[Aviso] {filepath} sem dados válidos.")
        else:
            print(f"[Aviso] Arquivo não encontrado: {filepath}")

    if not node_data:
        print("\n[ERRO] Nenhum dado de reputação encontrado para plotar.")
        return

    print("\nGerando gráficos...")

    # Sempre gera o SVG vetorial e o HTML de visualização imediata
    generate_native_svg_and_html(node_data, args.output_dir, tau=args.tau)

    # Tenta gerar com Matplotlib (PNG/PDF/Janela)
    try:
        import matplotlib
        generate_matplotlib_plots(
            node_data,
            output_dir=args.output_dir,
            tau=args.tau,
            x_mode=args.x_mode,
            layout=args.layout,
            img_format=args.format,
            dpi=args.dpi,
            show=args.show
        )
    except ImportError:
        print("\n[Nota] Matplotlib não está instalado no ambiente Python atual.")
        print("       Os gráficos vetoriais (.svg) e o painel (.html) foram gerados nativamente com sucesso em:")
        print(f"       -> {os.path.abspath(args.output_dir)}")
        print("       Para gerar também em PNG/PDF com matplotlib, instale: pip install matplotlib")

    print("\n[Concluído] Processamento finalizado com sucesso!")


if __name__ == "__main__":
    main()
