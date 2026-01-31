import csv
import os
import threading
from datetime import datetime

LOG_HEADERS = [
    "timestamp", "rodada", "cluster", "classificador", "f1_score",
    "decisao", "conflito", "outlier", "centroid_distance"
]



class CounselorLogger:
    """Gerencia a escrita de logs em CSV de forma thread-safe."""

    def __init__(self, node_id, use_log_folder=True):
        self.node_id = node_id

        base_dir = os.path.join(os.path .dirname(os.path.abspath(__file__)), "..")
        self.log_dir = os.path.join(base_dir, "logs") if use_log_folder else base_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.conflitos_log_file = os.path.join(self.log_dir, f"{self.node_id}_conflitos_gerados.csv")
        self.conselhos_log_file = os.path.join(self.log_dir, f"{self.node_id}_conselhos_recebidos.csv")

        # NOVO: log em formato LONG para gráficos
        self.cluster_long_file = os.path.join(self.log_dir, f"{self.node_id}_cluster_classifier_f1_long.csv")

        self.lock = threading.Lock()

        self._init_log_file(self.conflitos_log_file, LOG_HEADERS)
        self._init_log_file(self.conselhos_log_file, LOG_HEADERS)

        # Cabeçalho LONG (pedido)
        long_headers = ["timestamp", "rodada", "cluster", "classificador", "f1_score", "decisao", "conflito", "outlier", "centroid_distance"]
        self._init_log_file(self.cluster_long_file, long_headers)

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

    # -------- NOVO (formato LONG pedido) --------
    def log_cluster_classifier_f1_long(self, rows):
        """
        rows: lista de dicts com chaves:
          - rodada, cluster, classificador, f1_score, decisao, conflito
        """
        ts = datetime.now().isoformat(timespec="seconds")

        with self.lock:
            try:
                with open(self.cluster_long_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for r in rows:
                        writer.writerow([
                            ts,
                            r.get("rodada"),
                            r.get("cluster"),
                            r.get("classificador"),
                            r.get("f1_score"),
                            r.get("decisao"),
                            r.get("conflito"),
                            r.get("outlier"),
                            r.get("centroid_distance"),
                        ])

            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.cluster_long_file}: {e}")
