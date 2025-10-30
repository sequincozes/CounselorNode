import json
import sys
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'peer_config.json')


class ConfigManager:
    """
    Gerencia o carregamento da lista de pares P2P e configurações de ML.
    Identifica o nó local com base na porta fornecida.
    """

    def __init__(self, local_port):
        self.config = self._load_config()
        self.local_port = local_port

        self.peers = self.config.get('counselor_peers', [])
        self.ml_config = self.config.get('ml_config', {})

        # Encontra as informações do nó local na lista de pares USANDO A PORTA
        self.local_info = self._find_local_info_by_port(self.local_port)

        if not self.local_info:
            raise ValueError(
                f"Nó com porta {local_port} não encontrado em 'counselor_peers' no peer_config.json"
            )

        # O IP local e o nome agora são lidos DIRETAMENTE do arquivo de configuração
        self.local_ip = self.local_info.get('ip')
        self.node_id = self.local_info.get('name', f'node_{local_port}')

    def _load_config(self):
        """Carrega o arquivo de configuração JSON."""
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERRO: Arquivo de configuração não encontrado em: {CONFIG_PATH}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERRO: Arquivo de configuração inválido.")
            sys.exit(1)

    def _find_local_info_by_port(self, port):
        """Encontra a entrada do nó local na lista de pares pela porta."""
        return next((p for p in self.peers if p['port'] == port), None)

    def get_local_info(self):
        """Retorna as informações do nó local (IP/Porta/Nome)."""
        return self.local_info

    def get_ml_config(self):
        """Retorna o dicionário de configuração de Machine Learning."""
        return self.ml_config

    def get_other_peers(self):
        """Encontra todos os pares que NÃO são este nó local."""
        # Compara pela porta, que deve ser única
        return [p for p in self.peers if p['port'] != self.local_port]