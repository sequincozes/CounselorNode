import json
import socket
import threading
import numpy as np
import time  # Importa 'time' para medir o processamento

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
        self.counseling_logic_fn = counseling_fn
        self.is_running = False

        # Injeção de dependência para Logging e obtenção do IP local
        self.logger = logger
        self.peer_manager = peer_manager

    def _handle_request(self, conn, addr):
        """Processa um pedido de aconselhamento recebido e LOGA o evento."""
        start_time = time.time()

        ip_origem = addr[0]
        ip_destino = self.peer_manager.get_local_info().get('ip', 'N/A')

        decision = "ERROR"
        requester_id = "Unknown"
        ground_truth = "N/A"

        try:
            data = conn.recv(BUFFER_SIZE).decode('utf-8')
            if not data: return

            request = json.loads(data)
            requester_id = request.get('requester_id', 'Unknown')
            amostra_str = request.get('amostra', 'N/A')
            ground_truth = request.get('ground_truth', 'N/A')  # Captura o ground truth

            # --- LÓGICA DE CLASSIFICAÇÃO REAL NO SERVIDOR ---
            try:
                amostra_array = np.array(json.loads(amostra_str), dtype=float)
                final_prediction_str = self.counseling_logic_fn(amostra_array)

                if final_prediction_str == '0':
                    decision = "NORMAL"
                    counsel_msg = "Análise concluída. Resultado: Tráfego normal (Alta Confiança - DCS)."
                elif final_prediction_str == 'UNKNOWN':
                    decision = "UNKNOWN"
                    counsel_msg = "Análise falhou: Cluster não mapeado no Nó Conselheiro."
                else:
                    decision = "INTRUSION"
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
        """Inicia o servidor em uma thread separada para escutar por pedidos P2P."""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.is_running = True

            display_host = self.host
            if display_host == '0.0.0.0':
                detected_ip = detect_local_ip()
                display_host = f"todas as interfaces (IP detectado: {detected_ip})"

            print(f"[SERVIDOR] Escutando por conexões em {display_host} na porta {self.port}...")

            def server_loop():
                while self.is_running:
                    try:
                        server_socket.settimeout(1)
                        conn, addr = server_socket.accept()
                        client_thread = threading.Thread(target=self._handle_request, args=(conn, addr))
                        client_thread.daemon = True
                        client_thread.start()
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.is_running:
                            print(f"[SERVIDOR] Erro no loop principal: {e}")
                        break

            server_thread = threading.Thread(target=server_loop)
            server_thread.daemon = True
            server_thread.start()

        except Exception as e:
            print(f"[SERVIDOR] Erro iniciando servidor: {e}")
            raise


class CounselorClient:
    """Implementa a funcionalidade de cliente para pedir conselho a outros pares."""

    def __init__(self, node_id, peer_manager, logger):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.logger = logger  # Injeção de dependência para Logging
        self.local_ip = self.peer_manager.get_local_info().get('ip', 'N/A')

    def request_counsel(self, sample_data_array, other_peers_list, ground_truth):
        """
        Seleciona um par aleatório, envia um pedido de aconselhamento e LOGA o evento.
        """
        print("\n--- PEDINDO ACONSELHAMENTO P2P ---")

        if not other_peers_list:
            print("[CLIENTE] Nenhum outro par disponível para pedir conselho.")
            return None

        # Seleciona um conselheiro aleatório
        target_peer = other_peers_list[np.random.randint(0, len(other_peers_list))]
        peer_ip = target_peer['ip']
        peer_port = target_peer['port']
        peer_name = target_peer['name']

        print(f"[CLIENTE] Conselheiro Selecionado: {peer_name} ({peer_ip}:{peer_port})")

        sample_data_str = json.dumps(sample_data_array.tolist())

        request_data = {
            "requester_id": self.node_id,
            "reason": "Conflito de classificador local",
            "amostra": sample_data_str,
            "ground_truth": str(ground_truth)  # Envia o ground truth para o log do servidor
        }
        request_message = json.dumps(request_data).encode('utf-8')

        start_time = time.time()
        response = None
        log_decision = "ERROR_CONNECTION"

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

        except socket.timeout:
            print(f"[CLIENTE] Erro: Timeout tentando comunicar com {peer_ip}:{peer_port}")
            log_decision = "ERROR_TIMEOUT"
        except ConnectionRefusedError:
            print(f"[CLIENTE] Erro: Conexão recusada. Conselheiro não está ativo.")
            log_decision = "ERROR_CONNECTION_REFUSED"
        except Exception as e:
            print(f"[CLIENTE] Erro na comunicação P2P: {e}")
            log_decision = f"ERROR: {type(e).__name__}"
        finally:
            if 'client_socket' in locals():
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
            # ---------------

        return response