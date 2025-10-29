import json
import socket
import threading
import numpy as np

BUFFER_SIZE = 1024


class CounselorServer:
    """Implementa o servidor que escuta por pedidos de aconselhamento P2P."""

    def __init__(self, host, port, node_id, counseling_fn):
        self.host = host
        self.port = port
        self.node_id = node_id
        # self.local_source_type foi REMOVIDO
        self.counseling_logic_fn = counseling_fn  # Função de callback para lógica ML
        self.is_running = False

    def _handle_request(self, conn, addr):
        """Processa um pedido de aconselhamento recebido."""
        try:
            data = conn.recv(BUFFER_SIZE).decode('utf-8')
            if not data: return

            request = json.loads(data)
            requester_id = request.get('requester_id', 'Unknown')
            amostra_str = request.get('amostra', 'N/A')

            # --- LÓGICA DE CLASSIFICAÇÃO REAL NO SERVIDOR ---
            try:
                # Converte a string de feature JSON de volta para um array NumPy
                amostra_array = np.array(json.loads(amostra_str), dtype=float)

                # Executa a lógica de classificação real (alta confiança)
                final_prediction_str = self.counseling_logic_fn(amostra_array)

                # Formata a resposta
                if final_prediction_str == '0':  # Exemplo: assumindo que a classe '0' é NORMAL
                    decision = "NORMAL"
                    counsel_msg = "Análise concluída. Resultado: Tráfego normal (Alta Confiança - DCS)."
                elif final_prediction_str == 'UNKNOWN':
                    decision = "UNKNOWN"
                    counsel_msg = "Análise falhou: Cluster não mapeado no Nó Conselheiro."
                else:  # Qualquer outra classe é INTRUSAO
                    decision = "INTRUSION"
                    counsel_msg = f"Intrusão Confirmada: Classe {final_prediction_str} (Alta Confiança via DCS)."

            except Exception as e:
                print(f"ERRO DE SERVIDOR: Classificação falhou: {e}")
                decision = "ERROR"
                counsel_msg = "Motor de classificação falhou nesta amostra."
            # ------------------------------------------------

            print(f"\n[SERVIDOR] Pedido de aconselhamento recebido de {requester_id} ({addr[0]}:{addr[1]})")

            response = {
                "counselor_id": self.node_id,
                "decision": decision,
                "counsel": counsel_msg
                # "source_type" foi REMOVIDO
            }

            conn.sendall(json.dumps(response).encode('utf-8'))
            print(f"[SERVIDOR] Conselho enviado: '{response['decision']}'.")

        except Exception as e:
            print(f"[SERVIDOR] Erro processando pedido: {e}")
        finally:
            conn.close()

    def start_listening(self):
        """Inicia o servidor em uma thread separada para escutar por pedidos P2P."""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.is_running = True

            print(f"[SERVIDOR] Escutando por conexões em {self.host}:{self.port}...")

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

    def __init__(self, node_id, peer_manager):
        self.node_id = node_id
        self.peer_manager = peer_manager

    def request_counsel(self, sample_data_array, other_peers_list):
        """
        Seleciona um par aleatório da lista fornecida e envia um pedido de aconselhamento.
        Aceita sample_data como um array NumPy.
        """
        print("\n--- PEDINDO ACONSELHAMENTO P2P ---")

        if not other_peers_list:
            print("[CLIENTE] Nenhum outro par disponível para pedir conselho.")
            return None

        # Seleciona um conselheiro aleatório da lista de outros pares
        target_peer = other_peers_list[np.random.randint(0, len(other_peers_list))]

        peer_ip = target_peer['ip']
        peer_port = target_peer['port']

        print(f"[CLIENTE] Conselheiro Selecionado: {target_peer['name']} ({peer_ip}:{peer_port})")

        # Converte o array NumPy (sample_data_array) para uma string JSON serializável
        sample_data_str = json.dumps(sample_data_array.tolist())

        request_data = {
            "requester_id": self.node_id,
            "reason": "Conflito de classificador local",
            "amostra": sample_data_str  # Enviado como uma string de array JSON
        }
        request_message = json.dumps(request_data).encode('utf-8')

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((peer_ip, peer_port))
            client_socket.sendall(request_message)

            response_data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
            response = json.loads(response_data)

            print("--- CONSELHO RECEBIDO ---")
            print(f"Decisão do Conselheiro ({response['counselor_id']}): {response['decision']}")
            print("--------------------------")
            return response

        except socket.timeout:
            print(f"[CLIENTE] Erro: Timeout tentando comunicar com {peer_ip}:{peer_port}")
        except ConnectionRefusedError:
            print(f"[CLIENTE] Erro: Conexão recusada. Conselheiro não está ativo.")
        except Exception as e:
            print(f"[CLIENTE] Erro na comunicação P2P: {e}")
        finally:
            if 'client_socket' in locals():
                client_socket.close()

        return None