import sys
import time
import numpy as np
import json

# Importando camadas
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient
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

        # Componente 2: Motor ML (Inteligência IDS)
        self.engine = ClassifierEngine(self.peer_manager.get_ml_config())  # Inicializa o treinamento DCS

        # Componente 3 & 4: Rede (O servidor recebe a função de classificação real)
        self.client = CounselorClient(self.node_id, self.peer_manager)
        self.server = CounselorServer(
            self.bind_host,  # Escuta em 0.0.0.0
            self.port,  # Usa a porta encontrada no config
            self.node_id,
            self._execute_counseling_logic  # Passa a lógica real do nó como callback
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

    def check_traffic_and_act(self, sample_data_array):
        """
        Processa uma amostra usando o DCS local e verifica por conflito.
        """
        # Exibe as primeiras 5 features da amostra
        print(f"\n[{self.node_id.upper()}] Analisando amostra (primeiras 5 features): {sample_data_array[:5]}...")

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
            counsel_response = self.client.request_counsel(sample_data_array, other_peers)

            if counsel_response and counsel_response['decision'] == 'INTRUSION':
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão Final: INTRUSION (Confirmado por {counsel_response['counselor_id']}).")
                # PONTO DE EXTENSÃO: Implementar Aprendizado Incremental
            else:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito resolvido ou sem conselho externo definitivo. Usando decisão local padrão.")
        else:
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {classification}.")