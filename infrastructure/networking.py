import json
import socket
import threading
import time
import os
import sys

# Aumentamos o buffer, pois pacotes de fofoca (várias amostras) 
# podem ser maiores que um pedido de conselho único (1024 bytes).
BUFFER_SIZE = 8192 

def detect_local_ip():
    """
    Tenta encontrar o IP local "principal" da máquina na rede.
    Para desenvolvimento local, retorna 127.0.0.1 para compatibilidade com peer_config.json
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        # Para desenvolvimento local, sempre usar 127.0.0.1
        if ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('172.'):
            ip = '127.0.0.1'
    except Exception:
        ip = '127.0.0.1'
    finally:
        if s:
            s.close()
    return ip

class GossipServer:
    """Implementa o servidor que escuta pacotes de conhecimento (Gossip) P2P."""

    def __init__(self, host, port, node_id, gossip_integration_fn, logger, peer_manager):
        self.host = host
        self.port = port
        self.node_id = node_id
        # Agora o callback não devolve uma decisão, apenas recebe os dados para aprender
        self.gossip_integration_fn = gossip_integration_fn 
        self.is_running = False

        self.logger = logger
        self.peer_manager = peer_manager

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

    def _recv_all(self, conn):
        """Função auxiliar para garantir que recebemos o JSON completo."""
        data = b""
        while True:
            part = conn.recv(BUFFER_SIZE)
            data += part
            if len(part) < BUFFER_SIZE:
                # Ou a mensagem terminou, ou não há mais dados no socket
                break
        return data.decode('utf-8')

    def _handle_request(self, conn, addr):
        """Processa um pacote de gossip recebido."""
        start_time = time.time()
        ip_origem = addr[0]

        try:
            data_str = self._recv_all(conn)
            if not data_str: return

            package = json.loads(data_str)
            msg_type = package.get('type', 'UNKNOWN')

            # Verifica se é um pacote de Gossip válido
            if msg_type == "GOSSIP_PUSH":
                node_origin = package.get('node_origin', 'Unknown')
                
                # --- LÓGICA DE INTEGRAÇÃO DE CONHECIMENTO ---
                try:
                    # Envia o pacote para o node.py integrar no modelo local
                    self.gossip_integration_fn(package)
                    status = "SUCCESS"
                    msg = "Conhecimento integrado com sucesso."
                except Exception as e:
                    print(f"[{self.node_id.upper()}] ERRO AO INTEGRAR GOSSIP: {e}")
                    status = "ERROR_INTEGRATION"
                    msg = str(e)
                
                print(f"[{self.node_id.upper()}] [SERVIDOR] Gossip recebido de {node_origin} ({ip_origem}:{addr[1]}).")

            else:
                status = "ERROR_BAD_REQUEST"
                msg = "Tipo de mensagem não suportado."

            # Devolve um ACK simples (Acknowledge)
            response = {
                "receiver_id": self.node_id,
                "status": status,
                "message": msg
            }
            conn.sendall(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"[{self.node_id.upper()}] [SERVIDOR] Erro processando conexão: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

            # Aqui você pode adaptar seu logger para registrar recebimentos de Gossip
            # em vez de "pedidos de conselho"
            # end_time = time.time()
            # processing_time_ms = (end_time - start_time) * 1000
            # self.logger.log_gossip_recebido(...)

    def start_listening(self):
        self.is_running = True
        self.server_socket.listen(5)
        print(f"[{self.node_id.upper()}] [SERVIDOR] Escutando Gossip em {self.host}:{self.port}...")

        def listen_thread():
            while self.is_running:
                try:
                    conn, addr = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_request, args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"[{self.node_id.upper()}] [SERVIDOR] Erro inesperado: {e}")
                    break

        threading.Thread(target=listen_thread, daemon=True).start()

    def stop_listening(self):
        self.is_running = False
        self.server_socket.close()


class GossipClient:
    """Implementa o cliente que envia (faz push) do conhecimento para outros pares."""

    def __init__(self, node_id, peer_manager, logger):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.logger = logger

    def send_gossip(self, target_peer, knowledge_package):
        """
        Envia amostras ou pesos do modelo para o vizinho selecionado.
        """
        peer_ip = target_peer['ip']
        peer_port = target_peer['port']
        peer_name = target_peer.get('name', f"{peer_ip}:{peer_port}")

        # Adiciona metadados de protocolo ao pacote
        knowledge_package["type"] = "GOSSIP_PUSH"
        knowledge_package["node_origin"] = self.node_id

        payload = json.dumps(knowledge_package).encode('utf-8')

        client_socket = None
        start_time = time.time()
        status = "ERROR_CONNECTION"

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((peer_ip, peer_port))
            
            # Envia todos os dados
            client_socket.sendall(payload)

            # Aguarda a confirmação (ACK)
            response_data = client_socket.recv(1024).decode('utf-8')
            response = json.loads(response_data)

            if response.get("status") == "SUCCESS":
                status = "SUCCESS"
                # Opcional: comentar o print abaixo em produção para não poluir o terminal
                # print(f"[{self.node_id.upper()}] [CLIENTE] Gossip aceito por {peer_name}.")
            else:
                status = response.get("status", "UNKNOWN_ERROR")
                print(f"[{self.node_id.upper()}] [CLIENTE] {peer_name} rejeitou o gossip: {response.get('message')}")

        except Exception as e:
            print(f"[{self.node_id.upper()}] [CLIENTE] Falha ao enviar gossip para {peer_name}: {e}")
        finally:
            if client_socket:
                client_socket.close()

            # Aqui você pode adaptar o seu logger para salvar o histórico de gossips enviados
            # self.logger.log_gossip_enviado(...)

        return status