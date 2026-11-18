"""
calculate_defense_confusion_matrix.py
=====================================
Calcula e exibe a matriz de confusão e métricas de desempenho dos desafios
emitidos pelo mecanismo de defesa (ReputationManager) na rede CounselorNode.

Os dados são lidos dos logs CSV gerados por cada nó:
  logs/<node_id>_reputacao.csv (eventos de desafio e reputação)
  ou opcionalmente logs/<node_id>_conflitos_gerados.csv

Uso:
----
  # Avalia todos os nós e gera relatório consolidado no terminal:
  python calculate_defense_confusion_matrix.py

  # Avalia apenas os desafios onde o par avaliado foi o no_3:
  python calculate_defense_confusion_matrix.py --peer no_3

  # Avalia apenas desafios emitidos pelo nó avaliador no_1:
  python calculate_defense_confusion_matrix.py --node no_1

  # Salva relatórios CSV na pasta de saída:
  python calculate_defense_confusion_matrix.py --save-csv --output-dir logs/metrics

  # Salva gráficos de calor (heatmap) PNG das matrizes de confusão:
  python calculate_defense_confusion_matrix.py --plot
"""

import os
import glob
import csv
import argparse
from collections import defaultdict
from datetime import datetime


def find_log_files(log_dir, from_conflitos=False, node_filter=None):
    """Encontra os arquivos de log no diretório."""
    suffix = "*_conflitos_gerados.csv" if from_conflitos else "*_reputacao.csv"
    pattern = os.path.join(log_dir, suffix)
    files = glob.glob(pattern)

    if node_filter:
        files = [f for f in files if os.path.basename(f).startswith(f"{node_filter}_")]

    return sorted(files)


def load_records_from_conflitos(filepath):
    """Lê registros de um arquivo *_conflitos_gerados.csv."""
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gabarito = row.get("ground_truth", "").strip()
            decisao = row.get("decisao", "").strip()

            if not gabarito or gabarito in ("N/A", "UNKNOWN", "N/A_from_peer", ""):
                continue

            records.append({
                "timestamp": row.get("timestamp"),
                "seq_event": None,
                "peer_name": row.get("name_conselheiro", "UNKNOWN"),
                "peer_ip": row.get("ip_destino"),
                "peer_port": None,
                "s_ij": 0,
                "f_ij": 0,
                "reputacao": 0.0,
                "estado": "N/A",
                "decisao_par": decisao,
                "gabarito": gabarito,
                "latencia_ms": float(row.get("tempo_de_processamento_ms", 0.0) or 0.0),
            })
    return records


def load_defense_challenges(filepath):
    """
    Lê um arquivo de log de reputação e extrai registros de desafios válidos.
    Retorna lista de dicionários com os campos relevantes.
    """
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Considera todos os tipos de eventos de desafio da defesa:
            # CHALLENGE (conflito normal), ACTIVE_CHALLENGE (proativo por amostras),
            # PROBE_HIT e PROBE_MISS (probes de comeback de nó isolado).
            tipo = row.get("tipo_evento", "CHALLENGE").strip().upper()
            if tipo not in {"CHALLENGE", "ACTIVE_CHALLENGE", "PROBE_HIT", "PROBE_MISS"}:
                continue

            gabarito = row.get("gabarito", "").strip()
            decisao = row.get("decisao_par", "").strip()

            # Descarta registros sem ground truth
            if not gabarito or gabarito in ("N/A", "UNKNOWN", "N/A_from_peer", ""):
                continue

            records.append({
                "timestamp": row.get("timestamp"),
                "seq_event": row.get("seq_event"),
                "peer_name": row.get("peer_name", "UNKNOWN"),
                "peer_ip": row.get("peer_ip"),
                "peer_port": row.get("peer_port"),
                "s_ij": int(row.get("s_ij", 0)),
                "f_ij": int(row.get("f_ij", 0)),
                "reputacao": float(row.get("reputacao", 0.0)),
                "estado": row.get("estado", "EVALUATING"),
                "decisao_par": decisao,
                "gabarito": gabarito,
                "latencia_ms": float(row.get("latencia_ms", 0.0) or 0.0),
            })
    return records


def compute_confusion_matrix(records, target_classes=None):
    """
    Calcula a matriz de confusão e métricas detalhadas a partir dos registros.

    Retorna:
      classes: lista de classes ordenadas
      matrix: dicionário {classe_real: {classe_predita: contagem}}
      metrics: métricas por classe e agregadas
      errors: contagem de decisões com erro ou não-classificação
    """
    # Identifica todas as classes observadas
    real_classes = set()
    pred_classes = set()
    error_count = 0
    valid_records = []

    for r in records:
        gt = r["gabarito"].strip()
        dec = r["decisao_par"].strip()

        # Decisões de erro (ex: timeout, connection refused, LOOP_CLOSED)
        if dec.startswith("ERROR") or dec in ("LOOP_CLOSED", "N/A", "UNKNOWN", ""):
            error_count += 1
            continue

        real_classes.add(gt)
        pred_classes.add(dec)
        valid_records.append((gt, dec))

    if target_classes:
        all_classes = sorted(list(set(target_classes) | real_classes | pred_classes))
    else:
        # Coloca 'benign' primeiro se existir, seguido das outras em ordem alfabética
        all_classes = sorted(list(real_classes | pred_classes))
        if "benign" in all_classes:
            all_classes.remove("benign")
            all_classes.insert(0, "benign")

    # Inicializa matriz com zeros
    matrix = {r_cls: {p_cls: 0 for p_cls in all_classes} for r_cls in all_classes}

    for gt, dec in valid_records:
        if gt in matrix and dec in matrix[gt]:
            matrix[gt][dec] += 1

    total_valid = len(valid_records)

    # Cálculo de métricas por classe
    class_metrics = {}
    sum_tp = 0

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

    # Métricas globais
    accuracy = sum_tp / total_valid if total_valid > 0 else 0.0

    # Macro averages (média simples entre classes com suporte > 0)
    active_classes = [c for c in all_classes if class_metrics[c]["support"] > 0]
    n_active = len(active_classes) if active_classes else 1

    macro_precision = sum(class_metrics[c]["precision"] for c in active_classes) / n_active
    macro_recall = sum(class_metrics[c]["recall"] for c in active_classes) / n_active
    macro_f1 = sum(class_metrics[c]["f1_score"] for c in active_classes) / n_active

    # Weighted averages
    if total_valid > 0:
        weighted_precision = sum(class_metrics[c]["precision"] * class_metrics[c]["support"] for c in all_classes) / total_valid
        weighted_recall = sum(class_metrics[c]["recall"] * class_metrics[c]["support"] for c in all_classes) / total_valid
        weighted_f1 = sum(class_metrics[c]["f1_score"] * class_metrics[c]["support"] for c in all_classes) / total_valid
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    summary_metrics = {
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
    }

    return all_classes, matrix, class_metrics, summary_metrics


def format_matrix_table(classes, matrix):
    """Formata a matriz de confusão como uma tabela em texto."""
    if not classes:
        return "(Nenhuma amostra para exibir)"

    max_cls_len = max(len(c) for c in classes)
    col_width = max(max_cls_len, 10)

    # Cabeçalho
    header_title = f"{'Classe Real':>{col_width}} | " + " ".join(f"{c:>{col_width}}" for c in classes) + " | Total"
    sep = "-" * len(header_title)

    lines = [sep, header_title, sep]

    for r_cls in classes:
        row_counts = [matrix[r_cls][p_cls] for p_cls in classes]
        row_total = sum(row_counts)
        counts_str = " ".join(f"{cnt:>{col_width}}" for cnt in row_counts)
        lines.append(f"{r_cls:>{col_width}} | {counts_str} | {row_total:>{col_width}}")

    lines.append(sep)
    # Totais por coluna
    col_totals = [sum(matrix[r_cls][p_cls] for r_cls in classes) for p_cls in classes]
    total_all = sum(col_totals)
    col_totals_str = " ".join(f"{cnt:>{col_width}}" for cnt in col_totals)
    lines.append(f"{'Total Pred':>{col_width}} | {col_totals_str} | {total_all:>{col_width}}")
    lines.append(sep)

    return "\n".join(lines)


def format_metrics_table(classes, class_metrics, summary):
    """Formata as métricas por classe e resumo global."""
    if not classes:
        if summary.get("error_samples", 0) > 0:
            return (
                f"Aviso: Todas as {summary['error_samples']} requisições falharam "
                f"com erros de comunicação/rede (ex: ConnectionRefused, Timeout).\n"
                f"Nenhuma predição válida de classe foi recebida."
            )
        return ""

    lines = []
    lines.append(f"{'Classe':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Especif.':<10} | {'Support':<8}")
    lines.append("-" * 78)

    for c in classes:
        m = class_metrics[c]
        lines.append(
            f"{c:<18} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | "
            f"{m['f1_score']:<10.4f} | {m['specificity']:<10.4f} | {m['support']:<8d}"
        )

    lines.append("-" * 78)
    lines.append(f"Acurácia Global   : {summary['accuracy']:.4f} ({summary['valid_samples']} amostras avaliadas)")
    lines.append(f"Macro F1-Score    : {summary['macro_f1']:.4f}")
    lines.append(f"Weighted F1-Score : {summary['weighted_f1']:.4f}")
    if summary["error_samples"] > 0:
        lines.append(f"Erros de Rede/Out : {summary['error_samples']} requisições falharam")

    return "\n".join(lines)


def export_to_csv(classes, matrix, class_metrics, summary, output_path):
    """Salva a matriz de confusão e métricas em formato CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# MATRIZ DE CONFUSAO (Linhas: Real, Colunas: Predito)"])
        writer.writerow(["Classe_Real"] + classes + ["Total_Real"])
        for r_cls in classes:
            row_counts = [matrix[r_cls][p_cls] for p_cls in classes]
            writer.writerow([r_cls] + row_counts + [sum(row_counts)])
        writer.writerow([])

        writer.writerow(["# METRICAS POR CLASSE"])
        writer.writerow(["Classe", "Precision", "Recall", "F1_Score", "Especificidade", "TP", "FP", "FN", "TN", "Support"])
        for c in classes:
            m = class_metrics[c]
            writer.writerow([
                c, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1_score']:.4f}",
                f"{m['specificity']:.4f}", m["TP"], m["FP"], m["FN"], m["TN"], m["support"]
            ])
        writer.writerow([])

        writer.writerow(["# RESUMO GLOBAL"])
        writer.writerow(["Metrica", "Valor"])
        for k, v in summary.items():
            val = f"{v:.4f}" if isinstance(v, float) else v
            writer.writerow([k, val])


def try_plot_confusion_matrix(classes, matrix, title, output_png):
    """Gera imagem PNG com mapa de calor se matplotlib/seaborn estiverem disponíveis."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.array([[matrix[r][c] for c in classes] for r in classes])
        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), max(5, len(classes) * 1.0)))

        cax = ax.matshow(data, cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="left")
        ax.set_yticklabels(classes)

        ax.set_xlabel("Decisão do Conselheiro (Predição)", fontweight="bold")
        ax.set_ylabel("Gabarito (Ground Truth)", fontweight="bold")
        ax.set_title(title, pad=20, fontweight="bold")

        # Texto nos blocos
        thresh = data.max() / 2.0 if data.max() > 0 else 1.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = data[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"[Grafico Salvo] {output_png}")
    except ImportError:
        pass
    except Exception as e:
        print(f"[Aviso] Nao foi possivel gerar grafico PNG: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Calcula a matriz de confusão e métricas dos desafios emitidos pelo mecanismo de defesa."
    )
    parser.add_argument(
        "--from-conflitos",
        action="store_true",
        help="Lê também os logs de conselhos solicitados (*_conflitos_gerados.csv) em vez de apenas reputação"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Diretório onde estão os logs de reputação (padrão: logs)"
    )
    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="Filtra os desafios emitidos por um nó avaliador específico (ex: no_1)"
    )
    parser.add_argument(
        "--peer",
        type=str,
        default=None,
        help="Filtra os desafios respondidos por um conselheiro desafiado específico (ex: no_3)"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Salva as tabelas de métricas e matrizes de confusão em arquivos CSV"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Gera imagens PNG das matrizes de confusão (requer matplotlib)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/defense_metrics",
        help="Diretório de saída para relatórios e gráficos (padrão: logs/defense_metrics)"
    )

    args = parser.parse_args()

    # 1. Carrega logs
    log_files = find_log_files(args.log_dir, from_conflitos=args.from_conflitos, node_filter=args.node)
    if not log_files:
        tipo_log = "*_conflitos_gerados.csv" if args.from_conflitos else "*_reputacao.csv"
        print(f"[Aviso] Nenhum arquivo {tipo_log} encontrado em '{args.log_dir}'"
              f"{f' para nó {args.node}' if args.node else ''}.")
        return

    titulo = "CONSELHOS SOLICITADOS (CONFLITOS GERADOS)" if args.from_conflitos else "DESAFIOS DO MECANISMO DE DEFESA (REPUTATION MANAGER)"
    print("=" * 80)
    print(f"  AVALIAÇÃO DE {titulo}")
    print("=" * 80)
    print(f"Arquivos de log analisados: {len(log_files)}")
    for f in log_files:
        print(f" - {f}")
    print()

    all_records = []
    records_by_evaluator = defaultdict(list)
    loader_fn = load_records_from_conflitos if args.from_conflitos else load_defense_challenges

    suffix_clean = "_conflitos_gerados.csv" if args.from_conflitos else "_reputacao.csv"
    for f in log_files:
        evaluator_node = os.path.basename(f).replace(suffix_clean, "")
        recs = loader_fn(f)
        for r in recs:
            r["evaluator_node"] = evaluator_node
        records_by_evaluator[evaluator_node].extend(recs)
        all_records.extend(recs)

    if not all_records:
        print("[Info] Os arquivos de reputação foram encontrados, mas ainda não contêm desafios registrados.")
        print("       Dispare amostras aos nós para gerar desafios na rede antes de executar a análise.")
        return

    # Filtra por conselheiro se solicitado
    if args.peer:
        all_records = [r for r in all_records if r["peer_name"] == args.peer]
        if not all_records:
            print(f"[Aviso] Nenhum desafio encontrado para o par '{args.peer}'.")
            return

    # 2. Agrupamento por conselheiro avaliado (peer_name)
    records_by_peer = defaultdict(list)
    for r in all_records:
        records_by_peer[r["peer_name"]].append(r)

    # 3. Análise detalhada por Conselheiro Desafiado
    for peer_name, p_records in sorted(records_by_peer.items()):
        # Último estado e reputação do par
        last_state = p_records[-1]["estado"] if p_records else "N/A"
        last_rep = p_records[-1]["reputacao"] if p_records else 0.0
        s_tot = p_records[-1]["s_ij"] if p_records else 0
        f_tot = p_records[-1]["f_ij"] if p_records else 0

        print("\n" + "=" * 80)
        print(f"  DESEMPENHO DO CONSELHEIRO DESAFIADO: {peer_name.upper()}")
        print(f"  Total de desafios: {len(p_records)} | Acertos: {s_tot} | Falhas: {f_tot}")
        print(f"  Reputação Final: {last_rep:.4f} | Estado na Defesa: {last_state}")
        print("=" * 80)

        classes, matrix, class_metrics, summary = compute_confusion_matrix(p_records)

        print("\n[ Matriz de Confusão ] (Linhas = Gabarito Real, Colunas = Decisão do Nó)")
        print(format_matrix_table(classes, matrix))

        print("\n[ Métricas de Classificação ]")
        print(format_metrics_table(classes, class_metrics, summary))

        if args.save_csv:
            out_file = os.path.join(args.output_dir, f"cm_conselheiro_{peer_name}.csv")
            export_to_csv(classes, matrix, class_metrics, summary, out_file)
            print(f"\n[Exportado CSV] {out_file}")

        if args.plot:
            png_file = os.path.join(args.output_dir, f"cm_conselheiro_{peer_name}.png")
            try_plot_confusion_matrix(classes, matrix, f"Matriz de Confusão — Desafios do {peer_name.upper()}", png_file)

    # 4. Análise Global Consolidada da Rede
    if len(records_by_peer) > 1 and not args.peer:
        print("\n" + "=" * 80)
        print("  CONSOLIDADO GERAL DA REDE DE DEFESA (TODOS OS DESAFIOS)")
        print("=" * 80)
        classes, matrix, class_metrics, summary = compute_confusion_matrix(all_records)

        print("\n[ Matriz de Confusão Consolidada ]")
        print(format_matrix_table(classes, matrix))

        print("\n[ Métricas Consolidadas ]")
        print(format_metrics_table(classes, class_metrics, summary))

        if args.save_csv:
            out_file = os.path.join(args.output_dir, "cm_consolidado_defesa.csv")
            export_to_csv(classes, matrix, class_metrics, summary, out_file)
            print(f"\n[Exportado CSV] {out_file}")

        if args.plot:
            png_file = os.path.join(args.output_dir, "cm_consolidado_defesa.png")
            try_plot_confusion_matrix(classes, matrix, "Matriz de Confusão Consolidada — Defesa", png_file)

    print("\n[Concluído] Processamento finalizado com sucesso.\n")


if __name__ == "__main__":
    main()
