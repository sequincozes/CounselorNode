import sys
import time

# Importação da camada de infraestrutura
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient


class CounselorNode:
    """A classe principal que representa o IDS (Detector) na Counselors Network."""

    def __init__(self, node_id):
        self.node_id = node_id
        print(f"Node ID: {node_id}")

        # Componente 1: Configuração e Peers
        self.peer_manager = ConfigManager(node_id)
        local_info = self.peer_manager.get_local_info()
        self.host = local_info['ip']
        self.port = local_info['port']
        self.source_type = local_info['source_type']

        # Componente 2 & 3: Rede
        self.client = CounselorClient(node_id, self.peer_manager)
        self.server = CounselorServer(self.host, self.port, self.node_id, self.source_type)

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"Endereço: {self.host}:{self.port}")
        print(f"Especialidade: {self.source_type}")

    def start(self):
        """Inicia o servidor e mantém o nó ativo."""
        self.server.start_listening()

    def check_traffic_and_act(self, sample):
        """
        Simula o motor de detecção local (IDS) que decide se precisa de um conselho.
        """
        print(f"\n[{self.node_id.upper()}] Analisando amostra: {sample['data'][:20]}...")

        # Lógica de Gatilho (Conflito de Classificadores)
        if "login_attempts:high" in sample['data'] and self.source_type == 'conexao_tcp':
            print(f"[{self.node_id.upper()}] Alerta: Conflito de classificação local! Recorrendo à Counselors Network.")

            required_type = 'log_aplicacao'  # Busca o especialista
            counsel_response = self.client.request_counsel(sample['data'], required_type)

            if counsel_response and counsel_response['decision'] == 'INTRUSAO':
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão final: INTRUSAO (Confirmado por {counsel_response['counselor_id']}).")
            else:
                print(f"[{self.node_id.upper()}] Ação: Conflito resolvido. Usando classificação padrão.")
        else:
            print(f"[{self.node_id.upper()}] Classificação local: Tráfego normal.")