import sys
import time
import numpy as np
import json

# Importando camadas
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient
from infrastructure.logger import CounselorLogger  # Importa o novo logger
from core.classifier_engine import ClassifierEngine


class CounselorNode:
    """A classe principal que representa o IDS (Detector) na Counselors Network."""

    def __init__(self, detected_ip):
        # Componente 1: Configuração (Inicializado com o IP detectado)
        self.peer_manager = ConfigManager(detected_ip)
        local_info = self.peer_manager.get_local_info()

        # Informações lidas do config
        self.port = local_info.get('port')
        self.node_id = local_info.get('name')
        self.config_ip = local_info.get('ip')  # O IP que outros pares usam para ME encontrar

        # O host de "bind" é 0.0.0.0 para escutar em todas as interfaces
        self.bind_host = '0.0.0.0'

        # Componente 1.5: Logger
        # O logger é instanciado aqui e injetado nos componentes de rede
        self.logger = CounselorLogger(self.node_id)
        print(f"Logger inicializado. Logs serão salvos em 'logs/{self.node_id}_*.csv'")

        # Componente 2: Motor ML (Inteligência IDS)
        self.engine = ClassifierEngine(self.peer_manager.get_ml_config())  # Inicializa o treinamento DCS

        # Componente 3 & 4: Rede (Passa o logger e o peer_manager)
        self.client = CounselorClient(self.node_id, self.peer_manager, self.logger)
        self.server = CounselorServer(
            self.bind_host,  # Escuta em 0.0.0.0
            self.port,  # Usa a porta encontrada no config
            self.node_id,
            self._execute_counseling_logic,  # Passa a lógica real do nó como callback
            self.logger,  # Injeta o logger
            self.peer_manager  # Injeta o peer_manager para IP local
        )

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"IP de Configuração (Encontrado via {detected_ip}): {self.config_ip}:{self.port}")
        print(f"Endereço de Bind (Escutando em):             {self.bind_host}:{self.port}")
        print("-" * 30)

    def start(self):
        """Inicia o servidor e mantém o nó ativo."""
        self.server.start_listening()

    def _execute_counseling_logic(self, sample_data_array):
        """
        Função de callback executada quando este nó recebe um pedido de aconselhamento.
        """
        return self.engine.counseling_logic(sample_data_array)

    def check_traffic_and_act(self, sample_data_array, ground_truth):
        """
        Processa uma amostra usando o DCS local e verifica por conflito.
        Agora aceita 'ground_truth' para logging.
        """
        # Exibe as primeiras 5 features da amostra
        print(f"\n[{self.node_id.upper()}] Analisando amostra (primeiras 5 features): {sample_data_array[:5]}...")
        print(f"[{self.node_id.upper()}] (Ground Truth para esta amostra: {ground_truth})")

        # 1. Classifica e verifica por conflito usando o motor DCS
        results = self.engine.classify_and_check_conflict(sample_data_array)

        classification = results['classification']
        conflict = results['conflict']
        decisions = results['decisions']
        cluster_id = results.get('cluster_id', 'N/A')

        print(f"[{self.node_id.upper()}] Resultado DCS Local (Cluster {cluster_id}): {classification}")
        print(f"[{self.node_id.upper()}] Decisões Locais (Classes): {decisions}")

        # 2. Lógica de Gatilho da Rede de Conselheiros
        if conflict:
            print(
                f"[{self.node_id.upper()}] Alerta: CONFLITO DE CLASSIFICADOR DETECTADO! Consultando a Counselors Network.")

            # Busca outros pares para consultar
            other_peers = self.peer_manager.get_other_peers()

            if not other_peers:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito detectado, mas não há outros pares para consultar. Usando decisão local padrão.")
                return

            # O cliente agora selecionará um par aleatório da lista
            # Passa o ground_truth para o cliente para fins de logging
            counsel_response = self.client.request_counsel(
                sample_data_array,
                other_peers,
                ground_truth
            )

            if counsel_response and counsel_response['decision'] == 'INTRUSION':
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão Final: INTRUSION (Confirmado por {counsel_response['counselor_id']}).")
            else:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito resolvido ou sem conselho externo definitivo. Usando decisão local padrão.")
        else:
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {classification}.")