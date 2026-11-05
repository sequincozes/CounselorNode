import json
import socket
import threading
import numpy as np
import time
import os
import sys

# Aumentado para suportar vetores de features (25+ floats) em JSON
BUFFER_SIZE = 65536


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


def _recv_all(sock, buffer_size=BUFFER_SIZE):
    """
    Lê todos os dados disponíveis no socket até que a conexão seja fechada
    ou não haja mais dados. Necessário para mensagens maiores que BUFFER_SIZE.
    """
    chunks = []
    while True:
        try:
            chunk = sock.recv(buffer_size)
            if not chunk:
                break
            chunks.append(chunk)
            # Tenta decodificar: se for JSON válido, terminou
            try:
                json.loads(b"".join(chunks).decode('utf-8'))
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Ainda incompleto, continua lendo
                continue
        except socket.timeout:
            break
    return b"".join(chunks)


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
            raw_data = _recv_all(conn)
            if not raw_data:
                return

            data = raw_data.decode('utf-8')
            request = json.loads(data)
            requester_id = request.get('requester_id', 'Unknown')
            amostra_str = request.get('amostra', 'N/A')
            ground_truth = request.get('ground_truth', 'N/A')
            requester_chain = request.get('requester_chain', [])

            # --- LÓGICA DE CLASSIFICAÇÃO REAL NO SERVIDOR ---
            try:
                amostra_array = np.array(json.loads(amostra_str), dtype=float)

                # O callback agora recebe todos os argumentos necessários
                final_prediction_str = self.counseling_logic_fn(
                    amostra_array, requester_id, ip_origem, ground_truth, requester_chain
                )

                if final_prediction_str == 'LOOP_CLOSED':
                    decision = "LOOP_CLOSED"
                    counsel_msg = "Alerta: Loop de aconselhamento fechado. Ninguém tem a resposta. Decisão local de melhor modelo usada."
                else:
                    decision = final_prediction_str
                    counsel_msg = f"Intrusão Confirmada: Classe {final_prediction_str} (Alta Confiança via DCS)."

            except Exception as e:
                print(f"ERRO DE SERVIDOR: Classificação falhou: {e}")
                decision = "ERROR_CLASSIFICATION"
                counsel_msg = "Motor de classificação falhou nesta amostra."
            # ------------------------------------------------

            print(f"\n[SERVIDOR] Pedido de aconselhamento recebido de {requester_id} ({ip_origem}:{addr[1]})")

            end_time_inner = time.time()
            response = {
                "counselor_id":   self.node_id,
                "counselor_ip":   self.peer_manager.get_local_info().get('ip', ip_destino),
                "counselor_port": self.port,
                "decision":       decision,
                "counsel":        counsel_msg,
                "latencia_ms":    (end_time_inner - start_time) * 1000,
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
    """
    Implementa a funcionalidade de cliente para pedir conselho a outros pares.
    Integrado ao ReputationManager: filtra pares ISOLATED e usa o gt recebido
    para atualizar reputação quando disponível.
    """

    def __init__(self, node_id, peer_manager, logger, reputation_manager=None):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.logger = logger
        self.reputation_manager = reputation_manager  # Pode ser None (modo sem defesa)

        # BUSCA INFO LOCAL COMPLETA (IP e PORTA)
        local_info = self.peer_manager.get_local_info()
        self.local_ip = local_info.get('ip', '127.0.0.1')
        self.local_port = local_info.get('port')

    def request_counsel(self, sample_data_array, all_peers_list, ground_truth, requester_chain=None,
                        max_retries=5, retry_delay=3):
        """
        Filtra a lista para remover a si mesmo (IP + Porta), pares ISOLATED
        (via ReputationManager) e tenta contatar um conselheiro.
        Se o par sorteado ainda não estiver escutando (nó em boot),
        tenta novamente com outro par, respeitando um limite de tentativas.

        A resposta inclui counselor_ip e counselor_port para que o chamador
        possa atualizar a reputação do conselheiro.
        """
        print("\n--- PEDINDO ACONSELHAMENTO P2P ---")

        # Remove a si mesmo
        other_peers = [
            p for p in all_peers_list
            if not (p['ip'] == self.local_ip and p['port'] == self.local_port)
        ]

        # Filtra pares ISOLATED (se ReputationManager disponível)
        if self.reputation_manager is not None:
            before = len(other_peers)
            other_peers = self.reputation_manager.get_active_peers(other_peers)
            removed = before - len(other_peers)
            if removed > 0:
                print(f"[CLIENTE | {self.node_id}] (Defesa) {removed} par(es) ISOLATED removido(s) da lista de candidatos.")

        if not other_peers:
            print(f"[{self.node_id}] Alerta: Nenhum outro par disponível (Filtro resultou em lista vazia).")
            return None

        if requester_chain is None:
            requester_chain = []

        sample_data_str = json.dumps(sample_data_array.tolist())
        request_data = {
            "requester_id":    self.node_id,
            "reason":          "Conflito de classificador local",
            "amostra":         sample_data_str,
            "ground_truth":    str(ground_truth),
            "requester_chain": requester_chain
        }
        request_message = json.dumps(request_data).encode('utf-8')

        import random
        candidates = other_peers.copy()
        random.shuffle(candidates)

        for attempt in range(1, max_retries + 1):
            if not candidates:
                candidates = other_peers.copy()
                random.shuffle(candidates)

            target_peer = candidates.pop(0)
            peer_ip, peer_port, peer_name = target_peer['ip'], target_peer['port'], target_peer['name']

            print(f"[CLIENTE] Tentativa {attempt}/{max_retries}: conselheiro {peer_name} ({peer_ip}:{peer_port})")

            start_time = time.time()
            response, log_decision, client_socket = None, "ERROR_CONNECTION", None

            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(10)
                client_socket.connect((peer_ip, peer_port))
                client_socket.sendall(request_message)

                response_data = _recv_all(client_socket)
                response = json.loads(response_data.decode('utf-8'))
                log_decision = response.get('decision', 'ERROR_RESPONSE')

                print(f"--- CONSELHO RECEBIDO de {response['counselor_id']}: {log_decision} ---")

                # Garante que a resposta tenha campos de localização do conselheiro
                # (retrocompatível: se o servidor antigo não os enviar, usa target_peer)
                if 'counselor_ip' not in response:
                    response['counselor_ip'] = peer_ip
                if 'counselor_port' not in response:
                    response['counselor_port'] = peer_port

            except Exception as e:
                print(f"[CLIENTE] {peer_name} indisponível ({type(e).__name__}). "
                      f"Tentando novamente em {retry_delay}s...")
                log_decision = f"ERROR: {type(e).__name__}"
            finally:
                if client_socket:
                    client_socket.close()
                processing_time_ms = (time.time() - start_time) * 1000
                self.logger.log_conflito_gerado(
                    name_solicitante=self.node_id, name_conselheiro=peer_name,
                    ip_origem=self.local_ip, ip_destino=peer_ip,
                    tempo_proc_ms=processing_time_ms, decisao=log_decision, ground_truth=ground_truth
                )

            if response is not None:
                # Atualiza reputação aqui (nível de cliente, quando gt é conhecido)
                if self.reputation_manager is not None:
                    gt_str = str(ground_truth)
                    is_valid_gt = gt_str not in ("N/A", "N/A_from_peer", "UNKNOWN", "")
                    is_valid_dec = log_decision not in (
                        "ERROR_CONNECTION", "ERROR_TIMEOUT", "ERROR_CONNECTION_REFUSED",
                        "LOOP_CLOSED", "ERROR_CLASSIFICATION", "ERROR_RESPONSE"
                    ) and not log_decision.startswith("ERROR")

                    if is_valid_gt and is_valid_dec:
                        self.reputation_manager.register_challenge_result(
                            peer_ip=peer_ip,
                            peer_port=int(peer_port),
                            peer_decision=log_decision,
                            ground_truth=gt_str,
                            latencia_ms=processing_time_ms,
                        )

                return response

            time.sleep(retry_delay)

        print(f"[CLIENTE] Esgotadas {max_retries} tentativas. Nenhum conselheiro respondeu.")
        return None
