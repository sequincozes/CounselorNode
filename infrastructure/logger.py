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

# Header de decisões finais
DECISOES_HEADERS = [
    "timestamp",
    "ground_truth",
    "decisao",
    "responsavel"
]

AMOSTRAS_HEADERS = [
    "timestamp", "sample_id", "ground_truth", "decisao", "tempo_proc_ms"
]

# Mecanismo de defesa: eventos de reputação por desafio
REPUTACAO_HEADERS = [
    "timestamp",
    "seq_event",
    "peer_name",
    "peer_ip",
    "peer_port",
    "s_ij",
    "f_ij",
    "reputacao",
    "limiar_tau",
    "min_desafios",
    "estado",
    "tipo_evento",
    "decisao_par",
    "gabarito",
    "latencia_ms",
]

# Mecanismo de defesa: métricas agregadas
METRICAS_DEFESA_HEADERS = [
    "timestamp",
    "total_normal_requests",
    "total_challenges",
    "total_isolated_peers",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "precision",
    "recall",
    "false_positive_rate",
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

        # Decisões tomadas
        self.decisoes_file = os.path.join(self.log_dir, f"{self.node_id}_decisoes.csv")

        # Amostras recebidas
        self.amostras_log_file = os.path.join(self.log_dir, f"{self.node_id}_amostras_recebidas.csv")

        # Mecanismo de defesa
        self.reputacao_file    = os.path.join(self.log_dir, f"{self.node_id}_reputacao.csv")
        self.metricas_def_file = os.path.join(self.log_dir, f"{self.node_id}_metricas_defesa.csv")

        self.lock = threading.Lock()

        # Inicializa com headers corretos
        self._init_or_fix_header(self.conflitos_log_file,  CONSELHO_HEADERS)
        self._init_or_fix_header(self.conselhos_log_file,  CONSELHO_HEADERS)
        self._init_or_fix_header(self.cluster_long_file,   LONG_HEADERS)
        self._init_or_fix_header(self.decisoes_file,       DECISOES_HEADERS)
        self._init_or_fix_header(self.amostras_log_file,   AMOSTRAS_HEADERS)
        self._init_or_fix_header(self.reputacao_file,      REPUTACAO_HEADERS)
        self._init_or_fix_header(self.metricas_def_file,   METRICAS_DEFESA_HEADERS)

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

    # -------- Decisões tomadas --------
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

    # -------- Amostras recebidas --------
    def log_amostra_recebida(self, node_id, sample_id, ground_truth, decisao, tempo_proc_ms):
        with self.lock:
            with open(self.amostras_log_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), sample_id, ground_truth, decisao, tempo_proc_ms
                ])

    # -------- Mecanismo de Defesa: Reputação --------
    def log_reputacao(self, evento: dict):
        """
        Registra um evento de desafio/atualização de reputação.

        Parâmetros esperados no dict `evento`:
            seq_event, peer_name, peer_ip, peer_port,
            s_ij, f_ij, reputacao, limiar_tau, min_desafios,
            estado, tipo_evento, decisao_par, gabarito, latencia_ms
        """
        ts = datetime.now().isoformat(timespec="milliseconds")
        row = [
            ts,
            evento.get("seq_event", ""),
            evento.get("peer_name", ""),
            evento.get("peer_ip", ""),
            evento.get("peer_port", ""),
            evento.get("s_ij", ""),
            evento.get("f_ij", ""),
            f"{evento.get('reputacao', 0):.4f}",
            evento.get("limiar_tau", ""),
            evento.get("min_desafios", ""),
            evento.get("estado", ""),
            evento.get("tipo_evento", ""),
            evento.get("decisao_par", ""),
            evento.get("gabarito", ""),
            f"{evento.get('latencia_ms', 0):.2f}",
        ]
        with self.lock:
            try:
                with open(self.reputacao_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.reputacao_file}: {e}")

    # -------- Mecanismo de Defesa: Métricas Agregadas --------
    def log_metricas_defesa(self, metricas: dict):
        """
        Registra um snapshot das métricas agregadas do mecanismo de defesa.

        Parâmetros esperados no dict `metricas`:
            timestamp (str, opcional), total_normal_requests, total_challenges,
            total_isolated_peers, true_positives, false_positives,
            true_negatives, false_negatives, precision, recall, false_positive_rate
        """
        row = [
            metricas.get("timestamp", self._now()),
            metricas.get("total_normal_requests", 0),
            metricas.get("total_challenges", 0),
            metricas.get("total_isolated_peers", 0),
            metricas.get("true_positives", 0),
            metricas.get("false_positives", 0),
            metricas.get("true_negatives", 0),
            metricas.get("false_negatives", 0),
            f"{metricas.get('precision', 0.0):.4f}",
            f"{metricas.get('recall', 0.0):.4f}",
            f"{metricas.get('false_positive_rate', 0.0):.4f}",
        ]
        with self.lock:
            try:
                with open(self.metricas_def_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.metricas_def_file}: {e}")

    # -------- Snapshot F1 por cluster (legado) --------
    def log_cluster_f1_snapshot(self, node_id, rows, event="", sample_label="", counselor_id=""):
        """Mantido para compatibilidade com node.py::learn_with_advice."""
        ts = self._now()
        with self.lock:
            try:
                with open(self.cluster_long_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for r in rows:
                        writer.writerow([
                            ts,
                            r.get("cluster_id", ""),
                            r.get("max_f1", ""),
                            r.get("selected_models", ""),
                            r.get("below_min_f1", ""),
                            r.get("reason", ""),
                            r.get("n_train", ""),
                            r.get("n_eval", ""),
                            r.get("f1_by_classifier", ""),
                        ])
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {self.cluster_long_file}: {e}")
