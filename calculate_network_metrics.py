"""
calculate_network_metrics.py
============================
Calcula as métricas de desempenho globais da Rede de Conselhos como um todo:
  - Macro F1-Score
  - Acurácia Global (Accuracy)
  - Macro Precisão (Precision)
  - Macro Revogação / Sensibilidade (Recall)
  - Weighted F1-Score, Precisão e Recall ponderados
  - Matriz de Confusão Global da Rede
  - Desempenho individual comparativo de cada Nó (Nó 1, Nó 2, Nó 3)

Por padrão, analisa as decisões finais de todas as amostras processadas pela rede
(arquivos logs/*_amostras_recebidas.csv), permitindo também avaliar os desafios
do mecanismo de defesa (logs/*_reputacao.csv) ou conselhos solicitados.

Uso:
----
  # Avaliação padrão da rede consolidada:
  python calculate_network_metrics.py

  # Salva relatórios em CSV e texto na pasta de saída:
  python calculate_network_metrics.py --save-csv --save-report

  # Especifica diretório customizado de saída:
  python calculate_network_metrics.py --output-dir logs/network_metrics

  # Avalia a rede a partir dos desafios do mecanismo de defesa:
  python calculate_network_metrics.py --source challenges

  # Gera gráfico heatmap da matriz de confusão:
  python calculate_network_metrics.py --plot
"""

import os
import csv
import glob
import argparse
from collections import defaultdict, Counter


def load_sample_records(log_dir, source="samples", node_filter=None):
    """
    Carrega os registros de decisões dos nós da rede.

    Parâmetros:
    -----------
    log_dir : str
        Diretório onde estão os logs.
    source : str
        'samples'    -> logs/*_amostras_recebidas.csv (decisões finais do IDS no tráfego)
        'challenges' -> logs/*_reputacao.csv (desafios do ReputationManager)
        'conflitos'  -> logs/*_conflitos_gerados.csv (respostas de aconselhamento solicitadas)
    node_filter : str
        Filtro opcional para carregar apenas um nó específico (ex: 'no_1').

    Retorna:
    --------
    dict: { node_id: [ { 'ground_truth': str, 'decisao': str, 'latency_ms': float, ... } ] }
    """
    if source == "samples":
        pattern = os.path.join(log_dir, "*_amostras_recebidas.csv")
        suffix = "_amostras_recebidas.csv"
    elif source == "challenges":
        pattern = os.path.join(log_dir, "*_reputacao.csv")
        suffix = "_reputacao.csv"
    elif source == "conflitos":
        pattern = os.path.join(log_dir, "*_conflitos_gerados.csv")
        suffix = "_conflitos_gerados.csv"
    else:
        raise ValueError(f"Fonte desconhecida: {source}")

    files = glob.glob(pattern)
    if node_filter:
        files = [f for f in files if os.path.basename(f).startswith(f"{node_filter}_")]

    files = sorted(files)
    records_by_node = defaultdict(list)

    operational_labels = {
        "loop_closed", "error_connection", "error_timeout",
        "error_connection_refused", "error_classification",
        "error_response", "error", "n/a", "unknown", "none", ""
    }

    for filepath in files:
        node_id = os.path.basename(filepath).replace(suffix, "")

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if source == "samples":
                    gt = row.get("ground_truth", "").strip()
                    dec = row.get("decisao", "").strip()
                    lat = float(row.get("tempo_proc_ms", 0.0) or 0.0)
                elif source == "challenges":
                    gt = row.get("gabarito", "").strip()
                    dec = row.get("decisao_par", "").strip()
                    lat = float(row.get("latencia_ms", 0.0) or 0.0)
                elif source == "conflitos":
                    gt = row.get("ground_truth", "").strip()
                    dec = row.get("decisao", "").strip()
                    lat = float(row.get("tempo_de_processamento_ms", 0.0) or 0.0)

                # Ignora registros sem ground truth
                if not gt or gt in ("N/A", "UNKNOWN", "N/A_from_peer", ""):
                    continue

                is_op = (dec.lower() in operational_labels) or dec.lower().startswith("error")

                records_by_node[node_id].append({
                    "node_id": node_id,
                    "ground_truth": gt,
                    "decisao": dec,
                    "latency_ms": lat,
                    "is_operational": is_op,
                })

    return dict(records_by_node)


def compute_metrics(records):
    """
    Calcula matriz de confusão e métricas completas para um conjunto de registros:
      - Acurácia
      - Macro F1, Precisão e Recall
      - Weighted F1, Precisão e Recall
      - Métricas detalhadas por classe (TP, FP, FN, TN, Precision, Recall, F1, Specificity, Support)
    """
    if not records:
        return {}, {}, {}, []

    # Separa amostras válidas de classificação de eventuais falhas operacionais/erros de rede
    valid_records = [r for r in records if not r["is_operational"]]
    error_count = len(records) - len(valid_records)

    gt_set = set(r["ground_truth"] for r in valid_records)
    dec_set = set(r["decisao"] for r in valid_records)

    # Ordena classes colocando 'benign' em primeiro lugar, se existir
    all_classes = sorted(list(gt_set | dec_set))
    if "benign" in all_classes:
        all_classes.remove("benign")
        all_classes.insert(0, "benign")

    # Inicializa matriz de confusão
    matrix = {r_cls: {p_cls: 0 for p_cls in all_classes} for r_cls in all_classes}

    for r in valid_records:
        gt = r["ground_truth"]
        dec = r["decisao"]
        if gt in matrix and dec in matrix[gt]:
            matrix[gt][dec] += 1

    total_valid = len(valid_records)
    sum_tp = 0
    class_metrics = {}

    for cls in all_classes:
        tp = matrix[cls][cls]
        sum_tp += tp
        fn = sum(matrix[cls][p] for p in all_classes if p != cls)
        fp = sum(matrix[r][cls] for r in all_classes if r != cls)
        tn = total_valid - (tp + fn + fp)

        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        class_metrics[cls] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
        }

    accuracy = sum_tp / total_valid if total_valid > 0 else 0.0

    # Classes que efetivamente possuem amostras no Ground Truth
    target_classes = [c for c in all_classes if class_metrics[c]["support"] > 0]
    n_targets = len(target_classes) if target_classes else 1

    # Macro Averages (calculadas sobre as classes que existem no gabarito)
    macro_precision = sum(class_metrics[c]["precision"] for c in target_classes) / n_targets
    macro_recall = sum(class_metrics[c]["recall"] for c in target_classes) / n_targets
    macro_f1 = sum(class_metrics[c]["f1_score"] for c in target_classes) / n_targets

    # Weighted Averages (ponderadas pelo suporte de cada classe)
    if total_valid > 0:
        weighted_precision = sum(class_metrics[c]["precision"] * class_metrics[c]["support"] for c in all_classes) / total_valid
        weighted_recall = sum(class_metrics[c]["recall"] * class_metrics[c]["support"] for c in all_classes) / total_valid
        weighted_f1 = sum(class_metrics[c]["f1_score"] * class_metrics[c]["support"] for c in all_classes) / total_valid
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    # Latência média
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    summary = {
        "total_samples": len(records),
        "valid_samples": total_valid,
        "error_samples": error_count,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "avg_latency_ms": avg_latency,
    }

    return matrix, class_metrics, summary, all_classes


def format_matrix_table(classes, matrix):
    """Formata a matriz de confusão como uma tabela em texto de fácil leitura."""
    if not classes:
        return "(Sem amostras)"

    max_cls_len = max(len(c) for c in classes)
    col_width = max(max_cls_len, 11)

    header = f"{'Classe Real':>{col_width}} | " + " ".join(f"{c:>{col_width}}" for c in classes) + " | Total Real"
    sep = "=" * len(header)
    lines = [sep, header, "-" * len(header)]

    for r_cls in classes:
        counts = [matrix[r_cls][p_cls] for p_cls in classes]
        row_tot = sum(counts)
        counts_str = " ".join(f"{cnt:>{col_width}}" for cnt in counts)
        lines.append(f"{r_cls:>{col_width}} | {counts_str} | {row_tot:>{col_width}}")

    lines.append("-" * len(header))
    col_totals = [sum(matrix[r][c] for r in classes) for c in classes]
    tot_all = sum(col_totals)
    col_str = " ".join(f"{cnt:>{col_width}}" for cnt in col_totals)
    lines.append(f"{'Total Pred':>{col_width}} | {col_str} | {tot_all:>{col_width}}")
    lines.append(sep)

    return "\n".join(lines)


def format_class_metrics_table(classes, class_metrics):
    """Formata a tabela de métricas detalhadas por classe."""
    lines = []
    header = f"{'Classe':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Especif.':<10} | {'Support':<8}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for c in classes:
        m = class_metrics[c]
        lines.append(
            f"{c:<16} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | "
            f"{m['f1_score']:<10.4f} | {m['specificity']:<10.4f} | {m['support']:<8d}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_summary_card(summary, title="MÉTRICAS DA REDE DE CONSELHOS COMO UM TODO"):
    """Formata o cartão com as métricas consolidadas solicitadas."""
    lines = [
        "=" * 80,
        f"  {title}",
        "=" * 80,
        f"  Macro F1-Score       : {summary['macro_f1']:.4f}  ({summary['macro_f1']*100:.2f}%)",
        f"  Acurácia Global      : {summary['accuracy']:.4f}  ({summary['accuracy']*100:.2f}%)",
        f"  Macro Precisão       : {summary['macro_precision']:.4f}  ({summary['macro_precision']*100:.2f}%)",
        f"  Macro Revogação/Rec  : {summary['macro_recall']:.4f}  ({summary['macro_recall']*100:.2f}%)",
        "-" * 80,
        f"  Weighted F1-Score    : {summary['weighted_f1']:.4f}  ({summary['weighted_f1']*100:.2f}%)",
        f"  Total de Amostras    : {summary['total_samples']} (Válidas: {summary['valid_samples']} | Falhas: {summary['error_samples']})",
        f"  Tempo Médio Process. : {summary['avg_latency_ms']:.2f} ms por amostra",
        "=" * 80,
    ]
    return "\n".join(lines)


def format_nodes_comparison_table(node_summaries):
    """Formata uma tabela comparativa entre os nós individuais e a rede global."""
    lines = []
    header = f"{'Nó / Entidade':<16} | {'Acurácia':<10} | {'Macro F1':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Amostras':<8}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for name, s in node_summaries.items():
        lines.append(
            f"{name:<16} | {s['accuracy']:<10.4f} | {s['macro_f1']:<10.4f} | "
            f"{s['macro_precision']:<10.4f} | {s['macro_recall']:<10.4f} | {s['total_samples']:<8d}"
        )
    lines.append(sep)
    return "\n".join(lines)


def export_results_to_csv(output_dir, network_summary, class_metrics, matrix, classes, node_summaries):
    """Salva os relatórios em formato CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Resumo das métricas principais da rede
    sum_file = os.path.join(output_dir, "metricas_globais_rede.csv")
    with open(sum_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metrica", "Valor"])
        for k, v in network_summary.items():
            val = f"{v:.4f}" if isinstance(v, float) else v
            writer.writerow([k, val])
    print(f"[CSV Salvo] Métricas Globais: {sum_file}")

    # 2. Métricas por classe
    cls_file = os.path.join(output_dir, "metricas_por_classe.csv")
    with open(cls_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Classe", "Precision", "Recall", "F1_Score", "Especificidade", "TP", "FP", "FN", "TN", "Support"])
        for c in classes:
            m = class_metrics[c]
            writer.writerow([
                c, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1_score']:.4f}",
                f"{m['specificity']:.4f}", m["TP"], m["FP"], m["FN"], m["TN"], m["support"]
            ])
    print(f"[CSV Salvo] Métricas por Classe: {cls_file}")

    # 3. Matriz de confusão global
    cm_file = os.path.join(output_dir, "matriz_confusao_global.csv")
    with open(cm_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Classe_Real"] + classes + ["Total_Real"])
        for r_cls in classes:
            counts = [matrix[r_cls][p_cls] for p_cls in classes]
            writer.writerow([r_cls] + counts + [sum(counts)])
    print(f"[CSV Salvo] Matriz de Confusão Global: {cm_file}")

    # 4. Comparativo por nó
    cmp_file = os.path.join(output_dir, "comparativo_por_no.csv")
    with open(cmp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entidade", "Acuracia", "Macro_F1", "Macro_Precision", "Macro_Recall", "Total_Amostras"])
        for name, s in node_summaries.items():
            writer.writerow([
                name, f"{s['accuracy']:.4f}", f"{s['macro_f1']:.4f}",
                f"{s['macro_precision']:.4f}", f"{s['macro_recall']:.4f}", s["total_samples"]
            ])
    print(f"[CSV Salvo] Comparativo por Nó: {cmp_file}")


def try_plot_confusion_matrix(classes, matrix, title, output_png):
    """Plota a matriz de confusão consolidada em imagem PNG se matplotlib estiver disponível."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.array([[matrix[r][c] for c in classes] for r in classes])
        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.3), max(5, len(classes) * 1.1)), dpi=300)

        cax = ax.matshow(data, cmap="Blues")
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="left", fontsize=10, fontweight="bold")
        ax.set_yticklabels(classes, fontsize=10, fontweight="bold")

        ax.set_xlabel("Decisão Final da Rede de Conselhos", fontsize=11, fontweight="bold", labelpad=12)
        ax.set_ylabel("Gabarito Real (Ground Truth)", fontsize=11, fontweight="bold", labelpad=12)
        ax.set_title(title, pad=20, fontsize=13, fontweight="bold")

        thresh = data.max() / 2.0 if data.max() > 0 else 1.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = data[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:,}".replace(",", "."), ha="center", va="center", color=color, fontweight="bold", fontsize=11)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Gráfico Salvo] Matriz de Confusão: {output_png}")
    except ImportError:
        pass
    except Exception as e:
        print(f"[Aviso] Não foi possível gerar gráfico PNG com matplotlib: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Calcula Macro F1, Acurácia, Precisão e Recall da Rede de Conselhos como um todo."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Diretório onde estão os logs (padrão: logs)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="samples",
        choices=["samples", "challenges", "conflitos"],
        help="Fonte de dados para avaliação: 'samples' (*_amostras_recebidas.csv - padrão), 'challenges' (*_reputacao.csv) ou 'conflitos' (*_conflitos_gerados.csv)"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Salva as tabelas de métricas e matriz de confusão em arquivos CSV"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Salva o relatório completo em arquivo de texto na pasta de saída"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Gera imagem PNG da matriz de confusão global da rede"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/network_metrics",
        help="Diretório para salvar relatórios e figuras (padrão: logs/network_metrics)"
    )

    args = parser.parse_args()

    # 1. Carrega dados
    records_by_node = load_sample_records(args.log_dir, source=args.source)

    if not records_by_node:
        print(f"[ERRO] Nenhum registro encontrado em '{args.log_dir}' para a fonte '{args.source}'.")
        return

    # Consolida todos os registros da rede
    all_records = []
    for node_id, recs in sorted(records_by_node.items()):
        all_records.extend(recs)

    # 2. Calcula métricas da REDE COMO UM TODO
    net_matrix, net_class_metrics, net_summary, classes = compute_metrics(all_records)

    # 3. Calcula métricas individuais de cada Nó para comparação
    node_summaries = {}
    for node_id, recs in sorted(records_by_node.items()):
        _, _, n_summary, _ = compute_metrics(recs)
        node_summaries[f"Nó {node_id.replace('no_', '')} ({node_id})"] = n_summary

    # Adiciona a linha da Rede Global no comparativo
    node_summaries["REDE GLOBAL"] = net_summary

    # 4. Formata saídas
    card_text = format_summary_card(net_summary, "MÉTRICAS DA REDE DE CONSELHOS COMO UM TODO")
    matrix_text = format_matrix_table(classes, net_matrix)
    class_table_text = format_class_metrics_table(classes, net_class_metrics)
    comparison_text = format_nodes_comparison_table(node_summaries)

    # Imprime no terminal
    print("\n" + card_text)

    print("\n[ MATRIZ DE CONFUSÃO CONSOLIDADA DA REDE ] (Linhas: Gabarito Real | Colunas: Decisão da Rede)")
    print(matrix_text)

    print("\n[ DESEMPENHO DETALHADO POR CLASSE NA REDE ]")
    print(class_table_text)

    print("\n[ COMPARATIVO: DESEMPENHO INDIVIDUAL DOS NÓS vs REDE GLOBAL ]")
    print(comparison_text)
    print()

    # 5. Exportações opcionais
    if args.save_csv:
        export_results_to_csv(
            args.output_dir,
            net_summary,
            net_class_metrics,
            net_matrix,
            classes,
            node_summaries
        )

    if args.save_report:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, "relatorio_metricas_rede.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("RELATÓRIO DE DESEMPENHO — REDE DE CONSELHOS\n")
            f.write(f"Fonte dos dados: {args.source}\n\n")
            f.write(card_text + "\n\n")
            f.write("MATRIZ DE CONFUSÃO GLOBAL DA REDE:\n")
            f.write(matrix_text + "\n\n")
            f.write("MÉTRICAS POR CLASSE:\n")
            f.write(class_table_text + "\n\n")
            f.write("COMPARATIVO POR NÓ:\n")
            f.write(comparison_text + "\n")
        print(f"[Relatório Salvo] {report_path}")

    if args.plot:
        png_path = os.path.join(args.output_dir, "matriz_confusao_rede_global.png")
        try_plot_confusion_matrix(
            classes,
            net_matrix,
            "Matriz de Confusão Global da Rede de Conselhos",
            png_path
        )


if __name__ == "__main__":
    main()
