import json
import socket
import threading
import numpy as np
import time
import os
import sys

BUFFER_SIZE = 1024


def detect_local_ip():
    """
    Tenta encontrar o IP local "principal" da máquina na rede.
    Usa um truque comum de criar um socket e conectar-se a um IP externo.
    """
    s = None
    try:
        # Conecta a um IP externo (não envia dados)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'  # Fallback para localhost
    finally:
        if s:
            s.close()
    return ip


class CounselorServer:
    """Implementa o servidor que escuta por pedidos de aconselhamento P2P."""

    def __init__(self, host, port, node_id, counseling_fn, logger, peer_manager):
        self.host = host  # Será '0.0.0.0'
        self.port = port
        self.node_id = node_id
        # counseling_fn é o _execute_counseling_logic do node.py
        self.counseling_logic_fn = counseling_fn
        self.is_running = False

        self.logger = logger
        self.peer_manager = peer_manager

        # Cria o socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

    def _handle_request(self, conn, addr):
        """Processa um pedido de aconselhamento recebido e LOGA o evento."""
        start_time = time.time()

        ip_origem = addr[0]
        ip_destino = self.peer_manager.get_local_info().get('ip', 'N/A')

        decision = "ERROR"
        requester_id = "Unknown"
        ground_truth = "N/A"
        requester_chain = []  # Inicializa a cadeia de IPs

        try:
            data = conn.recv(BUFFER_SIZE).decode('utf-8')
            if not data: return

            request = json.loads(data)
            requester_id = request.get('requester_id', 'Unknown')
            amostra_str = request.get('amostra', 'N/A')
            ground_truth = request.get('ground_truth', 'N/A')
            # --- NOVO: Extrai a cadeia de IPs da requisição ---
            requester_chain = request.get('requester_chain', [])
            # ------------------------------------------------

            # --- LÓGICA DE CLASSIFICAÇÃO REAL NO SERVIDOR ---
            try:
                amostra_array = np.array(json.loads(amostra_str), dtype=float)

                # O callback agora recebe todos os argumentos necessários
                final_prediction_str = self.counseling_logic_fn(amostra_array, requester_id, ip_origem, ground_truth,
                                                                requester_chain)

                if final_prediction_str == 'LOOP_CLOSED':
                    decision = "LOOP_CLOSED"
                    counsel_msg = "Alerta: Loop de aconselhamento fechado. Ninguém tem a resposta. Decisão local de melhor modelo usada."
                # ------------------------------------
                else:
                    # Assumimos que qualquer outra coisa é uma classe de intrusão (ex: '1', '2')
                    decision = final_prediction_str
                    counsel_msg = f"Intrusão Confirmada: Classe {final_prediction_str} (Alta Confiança via DCS)."

            except Exception as e:
                print(f"ERRO DE SERVIDOR: Classificação falhou: {e}")
                decision = "ERROR_CLASSIFICATION"
                counsel_msg = "Motor de classificação falhou nesta amostra."
            # ------------------------------------------------

            print(f"\n[SERVIDOR] Pedido de aconselhamento recebido de {requester_id} ({ip_origem}:{addr[1]})")

            response = {
                "counselor_id": self.node_id,
                "decision": decision,
                "counsel": counsel_msg
            }

            conn.sendall(json.dumps(response).encode('utf-8'))
            print(f"[SERVIDOR] Conselho enviado: '{response['decision']}'.")

        except Exception as e:
            print(f"[SERVIDOR] Erro processando pedido: {e}")
            decision = f"ERROR_REQUEST: {type(e).__name__}"
        finally:
            if 'conn' in locals():
                conn.close()

            # --- LOGGING ---
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            self.logger.log_conselho_recebido(
                name_solicitante=requester_id,
                name_conselheiro=self.node_id,
                ip_origem=ip_origem,
                ip_destino=ip_destino,
                tempo_proc_ms=processing_time_ms,
                decisao=decision,
                ground_truth=ground_truth
            )
            # ---------------

    def start_listening(self):
        """Inicia o servidor em uma thread separada para não bloquear o nó principal."""
        self.is_running = True
        self.server_socket.listen(5)  # Permite 5 conexões enfileiradas

        print(f"[SERVIDOR] Escutando em {self.host}:{self.port}...")

        def listen_thread():
            while self.is_running:
                try:
                    conn, addr = self.server_socket.accept()
                    # Cria uma nova thread para lidar com o pedido
                    client_thread = threading.Thread(
                        target=self._handle_request, args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Apenas continua
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"[SERVIDOR] Erro inesperado: {e}")
                    break

        threading.Thread(target=listen_thread, daemon=True).start()

    def stop_listening(self):
        """Para o servidor."""
        self.is_running = False
        self.server_socket.close()


class CounselorClient:
    """Implementa a funcionalidade de cliente para pedir conselho a outros pares."""

    def __init__(self, node_id, peer_manager, logger):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.logger = logger

        # BUSCA INFO LOCAL COMPLETA (IP e PORTA)
        local_info = self.peer_manager.get_local_info()
        self.local_ip = local_info.get('ip', '127.0.0.1')
        self.local_port = local_info.get('port')

    def request_counsel(self, sample_data_array, all_peers_list, ground_truth, requester_chain=None):
        """
        Filtra a lista para remover a si mesmo (IP + Porta) e seleciona um par.
        """
        print("\n--- PEDINDO ACONSELHAMENTO P2P ---")

        # FILTRO INTELIGENTE: Essencial para rodar múltiplos nós no mesmo IP (127.0.0.1)
        other_peers = [
            p for p in all_peers_list
            if not (p['ip'] == self.local_ip and p['port'] == self.local_port)
        ]

        if not other_peers:
            print(f"[{self.node_id}] Alerta: Nenhum outro par disponível (Filtro resultou em lista vazia).")
            # Debug para entender quem estava na lista
            print(f"DEBUG: Eu sou {self.local_ip}:{self.local_port}. Lista total tinha {len(all_peers_list)} peers.")
            return None

        # 1. FIXANDO SEED PARA REPRODUTIBILIDADE
        import random
        # Usamos um seed fixo para garantir que o simulador escolha sempre o mesmo par no teste
        random.seed(42)
        target_peer = random.choice(other_peers)

        peer_ip = target_peer['ip']
        peer_port = target_peer['port']
        peer_name = target_peer['name']

        print(f"[CLIENTE] Conselheiro Selecionado: {peer_name} ({peer_ip}:{peer_port})")

        # --- PREPARAÇÃO DOS DADOS ---
        if requester_chain is None:
            requester_chain = []

        sample_data_str = json.dumps(sample_data_array.tolist())

        request_data = {
            "requester_id": self.node_id,
            "reason": "Conflito de classificador local",
            "amostra": sample_data_str,
            "ground_truth": str(ground_truth),
            "requester_chain": requester_chain
        }
        request_message = json.dumps(request_data).encode('utf-8')

        # --- COMUNICAÇÃO SOCKET ---
        start_time = time.time()
        response = None
        log_decision = "ERROR_CONNECTION"
        client_socket = None

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((peer_ip, peer_port))
            client_socket.sendall(request_message)

            response_data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
            response = json.loads(response_data)
            log_decision = response.get('decision', 'ERROR_RESPONSE')

            print("--- CONSELHO RECEBIDO ---")
            print(f"Decisão do Conselheiro ({response['counselor_id']}): {log_decision}")
            print("--------------------------")
            time.sleep(30)  # Espera entre as amostras
            print("DANDO UMA CALMADINHA")

        except Exception as e:
            print(f"[CLIENTE] Erro na comunicação P2P com {peer_name}: {e}")
            log_decision = f"ERROR: {type(e).__name__}"
        finally:
            if client_socket:
                client_socket.close()

            # --- LOGGING ---
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            self.logger.log_conflito_gerado(
                name_solicitante=self.node_id,
                name_conselheiro=peer_name,
                ip_origem=self.local_ip,
                ip_destino=peer_ip,
                tempo_proc_ms=processing_time_ms,
                decisao=log_decision,
                ground_truth=ground_truth
            )

        return response

