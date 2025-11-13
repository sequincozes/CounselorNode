import json
import sys
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'peer_config.json')


class ConfigManager:
    """
    Gerencia o carregamento da lista de pares P2P e configurações de ML.
    Identifica o nó local com base no IP detectado.
    """

    def __init__(self, local_ip):
        self.config = self._load_config()
        self.local_ip = local_ip

        self.peers = self.config.get('counselor_peers', [])
        self.ml_config = self.config.get('ml_config', {})

        # Encontra as informações do nó local na lista de pares USANDO O IP
        self.local_info = self._find_local_info_by_ip(self.local_ip)

        if not self.local_info:
            raise ValueError(
                f"Nó com IP {local_ip} não encontrado em 'counselor_peers' no peer_config.json"
            )

        # A porta local e o nome agora são lidos DIRETAMENTE do arquivo de configuração
        self.local_port = self.local_info.get('port')
        self.node_id = self.local_info.get('name', f'node_{local_ip}')

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

    def _find_local_info_by_ip(self, ip):
        """Encontra a entrada do nó local na lista de pares pelo IP."""
        # Assume que um IP é único para um nó.
        # Se você precisar executar vários nós na mesma máquina,
        # esta lógica precisaria ser mais complexa (IP + Porta).
        return next((p for p in self.peers if p['ip'] == ip), None)

    def get_local_info(self):
        """Retorna as informações do nó local (IP/Porta/Nome)."""
        return self.local_info

    def get_ml_config(self):
        """Retorna o dicionário de configuração de Machine Learning."""
        return self.ml_config

    def get_other_peers(self):
        """Encontra todos os pares que NÃO são este nó local."""
        # Filtra com base na combinação de IP e Porta
        return [p for p in self.peers if p['ip'] != self.local_ip ]