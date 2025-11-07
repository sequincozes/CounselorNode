import csv
import os
import threading
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

# Colunas para ambos os arquivos de log
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

    def __init__(self, node_id):
        self.node_id = node_id

        # Cria o diretório de logs se não existir
        os.makedirs(LOG_DIR, exist_ok=True)

        # Define os caminhos dos arquivos de log
        self.conflitos_log_file = os.path.join(LOG_DIR, f"{self.node_id}_conflitos_gerados.csv")
        self.conselhos_log_file = os.path.join(LOG_DIR, f"{self.node_id}_conselhos_recebidos.csv")

        self.lock = threading.Lock()

        # Inicializa os arquivos de log com cabeçalhos, se não existirem
        self._init_log_file(self.conflitos_log_file, LOG_HEADERS)
        self._init_log_file(self.conselhos_log_file, LOG_HEADERS)

    def _init_log_file(self, file_path, headers):
        """Cria o arquivo de log com cabeçalho se ele não existir."""
        with self.lock:
            if not os.path.exists(file_path):
                try:
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                except IOError as e:
                    print(f"ERRO DE LOG: Falha ao inicializar {file_path}: {e}")

    def _log_to_file(self, file_path, row_data):
        """Escreve uma linha em um arquivo CSV de forma thread-safe."""
        timestamp = datetime.now().isoformat()
        full_row = [timestamp] + row_data

        with self.lock:
            try:
                with open(file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(full_row)
            except IOError as e:
                print(f"ERRO DE LOG: Falha ao escrever em {file_path}: {e}")

    def log_conflito_gerado(self, name_solicitante, name_conselheiro, ip_origem, ip_destino, tempo_proc_ms, decisao,
                            ground_truth):
        """
        Loga um evento onde ESTE nó gerou um conflito (solicitou conselho).
        (Arquivo: conflitos_gerados.csv)
        """
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

    def log_conselho_recebido(self, name_solicitante, name_conselheiro, ip_origem, ip_destino, tempo_proc_ms, decisao,
                              ground_truth):
        """
        Loga um evento onde ESTE nó recebeu um pedido de conselho (agiu como conselheiro).
        (Arquivo: conselhos_recebidos.csv)
        """
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