import json
import sys
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'peer_config.json')


class ConfigManager:
    """
    Gerencia o carregamento da lista de pares P2P e configurações de ML.
    Identifica o nó local com base no IP detectado e na porta local.
    """

    def __init__(self, local_ip, local_port=None):
        self.config = self._load_config()
        self.local_ip = local_ip
        self.local_port = local_port

        self.peers = self.config.get('counselor_peers', [])
        self.ml_config = self.config.get('ml_config', {})

        # Localiza info do nó local
        if local_port:
            self.local_info = next(
                (p for p in self.peers if p['ip'] == local_ip and p['port'] == local_port),
                None
            )
            # Fallback: se o IP detectado for 127.0.0.1 mas no json estiver outro IP (ou vice-versa),
            # permite encontrar pela porta para facilitar testes na mesma máquina
            if not self.local_info:
                self.local_info = next(
                    (p for p in self.peers if p.get('port') == local_port),
                    None
                )
        else:
            self.local_info = self._find_local_info_by_ip(self.local_ip)

        if not self.local_info:
            raise ValueError(
                f"Nó com IP {local_ip} e porta {local_port} não encontrado no peer_config.json"
            )

        self.node_id = self.local_info.get('name', f'node_{local_ip}')
        self.local_port = self.local_info.get('port', local_port)

    def _find_local_info(self, ip, port):
        if port:
            return next((p for p in self.peers if p['ip'] == ip and p['port'] == port), None)
        return next((p for p in self.peers if p['ip'] == ip), None)

    def _load_config(self):
        """Carrega o arquivo de configuração JSON."""
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERRO: Arquivo de configuração não encontrado em: {CONFIG_PATH}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERRO: Arquivo de configuração inválido.")
            sys.exit(1)

    def _find_local_info_by_ip(self, ip):
        """Encontra a entrada do nó local na lista de pares pelo IP."""
        return next((p for p in self.peers if p['ip'] == ip), None)

    def get_local_info(self):
        """Retorna as informações do nó local (IP/Porta/Nome/Datasets)."""
        return self.local_info

    def get_ml_config(self):
        """
        Retorna o dicionário de configuração de Machine Learning.
        Se o nó local tiver configurações específicas de dataset no 'counselor_peers',
        elas sobrescrevem a configuração global padrão.
        """
        cfg = dict(self.ml_config)
        if self.local_info:
            for key in ["train_eval_dataset_source", "final_test_dataset_source", "target_column"]:
                if key in self.local_info:
                    cfg[key] = self.local_info[key]
        return cfg

    def get_other_peers(self):
        """Encontra todos os pares que NÃO são este nó local (compara IP e porta)."""
        my_port = self.local_port
        my_ip = self.local_ip
        my_name = self.node_id

        return [
            p for p in self.peers
            if not (
                (p.get('name') == my_name) or
                (p['port'] == my_port and (p['ip'] == my_ip or my_ip in ('127.0.0.1', 'localhost') or p['ip'] in ('127.0.0.1', 'localhost')))
            )
        ]
