import csv
import os
import threading
from datetime import datetime

LOG_HEADERS = [
    "timestamp",
    "name_solicitante",
    "name_conselheiro",
    "ip_origem",
    "ip_destino",
    "tempo_de_processamento_ms",
    "decisao",
    "ground_truth"
]


class CounselorLogger:
    """Gerencia a escrita de logs em CSV de forma thread-safe."""

    def __init__(self, node_id, use_log_folder=True):
        self.node_id = node_id

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.log_dir = os.path.join(base_dir, "logs") if use_log_folder else base_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.conflitos_log_file = os.path.join(self.log_dir, f"{self.node_id}_conflitos_gerados.csv")
        self.conselhos_log_file = os.path.join(self.log_dir, f"{self.node_id}_conselhos_recebidos.csv")

        # Novo arquivo: histórico de F1 por cluster
        self.cluster_f1_file = os.path.join(self.log_dir, f"{self.node_id}_cluster_f1_history.csv")

        self.lock = threading.Lock()

        self._init_log_file(self.conflitos_log_file, LOG_HEADERS)
        self._init_log_file(self.conselhos_log_file, LOG_HEADERS)

        cluster_headers = [
            "timestamp", "node_id", "event", "counselor_id", "sample_label",
            "cluster_id", "max_f1", "selected_models", "below_min_f1", "reason",
            "n_train", "n_eval", "f1_by_classifier"
        ]
        self._init_log_file(self.cluster_f1_file, cluster_headers)

    def _init_log_file(self, file_path, headers):
        with self.lock:
            if not os.path.exists(file_path):
                try:
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                except IOError as e:
                    print(f"ERRO DE LOG: Falha ao inicializar {file_path}: {e}")

    def _log_to_file(self, file_path, row_data):
        timestamp = datetime.now().isoformat()
        full_row = [timestamp] + row_data

        with self.lock:
            try:
                with open(file_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(full_row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {file_path}: {e}")

    def log_conflito_gerado(self, name_solicitante, name_conselheiro, ip_origem, ip_destino, tempo_proc_ms, decisao, ground_truth):
        row_data = [
            name_solicitante,
            name_conselheiro,
            ip_origem,
            ip_destino,
            f"{tempo_proc_ms:.4f}",
            decisao,
            str(ground_truth)
        ]
        self._log_to_file(self.conflitos_log_file, row_data)

    def log_conselho_recebido(self, name_solicitante, name_conselheiro, ip_origem, ip_destino, tempo_proc_ms, decisao, ground_truth):
        row_data = [
            name_solicitante,
            name_conselheiro,
            ip_origem,
            ip_destino,
            f"{tempo_proc_ms:.4f}",
            decisao,
            str(ground_truth)
        ]
        self._log_to_file(self.conselhos_log_file, row_data)

    def log_cluster_f1_snapshot(self, rows, event, sample_label, counselor_id="UNKNOWN"):
        ts = datetime.now().isoformat(timespec="seconds")

        with self.lock:
            try:
                with open(self.cluster_f1_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp", "node_id", "event", "counselor_id", "sample_label",
                            "cluster_id", "max_f1", "selected_models", "below_min_f1", "reason",
                            "n_train", "n_eval", "f1_by_classifier"
                        ],
                    )

                    for r in rows:
                        writer.writerow({
                            "timestamp": ts,
                            "node_id": self.node_id,
                            "event": event,
                            "counselor_id": counselor_id,
                            "sample_label": sample_label,
                            "cluster_id": r.get("cluster_id"),
                            "max_f1": r.get("max_f1"),
                            "selected_models": r.get("selected_models"),
                            "below_min_f1": r.get("below_min_f1"),
                            "reason": r.get("reason"),
                            "n_train": r.get("n_train"),
                            "n_eval": r.get("n_eval"),
                            "f1_by_classifier": r.get("f1_by_classifier"),
                        })
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.cluster_f1_file}: {e}")
