import csv
import os
import threading
from datetime import datetime

# Header dos logs antigos (conflitos/conselhos) — manter correto
CONSELHO_HEADERS = [
    "timestamp",
    "name_solicitante",
    "name_conselheiro",
    "ip_origem",
    "ip_destino",
    "tempo_de_processamento_ms",
    "decisao",
    "ground_truth",
]

# Header LONG para gráficos (com outlier e distância)
LONG_HEADERS = [
    "timestamp", "rodada", "cluster", "classificador", "f1_score",
    "decisao", "conflito", "outlier", "centroid_distance"
]

# NOVO: Header de decisões (pedido)
DECISOES_HEADERS = [
    "timestamp",
    "ground_truth",
    "decisao",
    "responsavel"
]


class CounselorLogger:
    """Gerencia a escrita de logs em CSV de forma thread-safe."""

    def __init__(self, node_id, use_log_folder=True):
        self.node_id = node_id

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.log_dir = os.path.join(base_dir, "logs") if use_log_folder else base_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Logs antigos
        self.conflitos_log_file = os.path.join(self.log_dir, f"{self.node_id}_conflitos_gerados.csv")
        self.conselhos_log_file = os.path.join(self.log_dir, f"{self.node_id}_conselhos_recebidos.csv")

        # LONG para gráficos
        self.cluster_long_file = os.path.join(self.log_dir, f"{self.node_id}_cluster_classifier_f1_long.csv")

        # NOVO: decisões tomadas
        self.decisoes_file = os.path.join(self.log_dir, f"{self.node_id}_decisoes.csv")

        self.lock = threading.Lock()

        # Inicializa com headers corretos
        self._init_or_fix_header(self.conflitos_log_file, CONSELHO_HEADERS)
        self._init_or_fix_header(self.conselhos_log_file, CONSELHO_HEADERS)
        self._init_or_fix_header(self.cluster_long_file, LONG_HEADERS)
        self._init_or_fix_header(self.decisoes_file, DECISOES_HEADERS)

    def _init_or_fix_header(self, file_path, expected_headers):
        """
        Cria arquivo se não existir.
        Se existir e header estiver diferente, reescreve apenas o header (zera o arquivo).
        """
        with self.lock:
            if not os.path.exists(file_path):
                try:
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(expected_headers)
                except IOError as e:
                    print(f"ERRO DE LOG: Falha ao inicializar {file_path}: {e}")
                return

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
            except IOError:
                first_line = ""

            expected_line = ",".join(expected_headers)
            if first_line != expected_line:
                try:
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(expected_headers)
                except IOError as e:
                    print(f"ERRO DE LOG: Falha ao corrigir cabeçalho de {file_path}: {e}")

    def _now(self):
        return datetime.now().isoformat(timespec="seconds")

    # -------- Logs antigos --------
    def log_conflito_gerado(self, name_solicitante, name_conselheiro, ip_origem, ip_destino, tempo_proc_ms, decisao, ground_truth):
        ts = self._now()
        row = [
            ts, name_solicitante, name_conselheiro, ip_origem, ip_destino,
            f"{tempo_proc_ms:.4f}", str(decisao), str(ground_truth)
        ]
        with self.lock:
            try:
                with open(self.conflitos_log_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.conflitos_log_file}: {e}")

    def log_conselho_recebido(self, name_solicitante, name_conselheiro, ip_origem, ip_destino, tempo_proc_ms, decisao, ground_truth):
        ts = self._now()
        row = [
            ts, name_solicitante, name_conselheiro, ip_origem, ip_destino,
            f"{tempo_proc_ms:.4f}", str(decisao), str(ground_truth)
        ]
        with self.lock:
            try:
                with open(self.conselhos_log_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.conselhos_log_file}: {e}")

    # -------- LONG para gráficos --------
    def log_cluster_classifier_f1_long(self, rows):
        ts = self._now()
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

    # -------- NOVO: decisões tomadas --------
    def log_decisao(self, ground_truth, decisao, responsavel):
        """
        Loga TODA decisão final tomada no simulador:
        timestamp, ground_truth, decisao, responsavel
        """
        ts = self._now()
        row = [ts, str(ground_truth), str(decisao), str(responsavel)]
        with self.lock:
            try:
                with open(self.decisoes_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.decisoes_file}: {e}")
