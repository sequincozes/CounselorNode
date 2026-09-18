# infrastructure/sample_networking.py
import json
import socket
import threading
import time
import numpy as np

RECV_CHUNK = 4096


class SampleReceiver:
    """
    Escuta amostras de tráfego enviadas pelo Sample Sender (processo externo à CN).
    Ao receber uma amostra, delega a classificação ao próprio nó via callback
    (tipicamente CounselorNode.check_traffic_and_act), reaproveitando toda a
    lógica de conflito/conselho já existente na rede de conselhos.
    """

    def __init__(self, host, port, node_id, traffic_fn, logger):
        self.host = host
        self.port = port
        self.node_id = node_id
        self.traffic_fn = traffic_fn  # callback: (sample_data_array, ground_truth) -> decisão
        self.logger = logger
        self.is_running = False

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

    def _recv_message(self, conn):
        """Lê até o delimitador '\\n', evitando o truncamento de payloads grandes
        que o BUFFER_SIZE fixo do canal de conselhos pode sofrer."""
        buffer = b""
        while b"\n" not in buffer:
            chunk = conn.recv(RECV_CHUNK)
            if not chunk:
                break
            buffer += chunk
        raw_message, _, _ = buffer.partition(b"\n")
        return raw_message.decode("utf-8")

    def _handle_sample(self, conn, addr):
        start_time = time.time()
        sample_id, ground_truth, decision = "N/A", "N/A", "ERROR"

        try:
            raw = self._recv_message(conn)
            if not raw:
                return

            message = json.loads(raw)
            sample_id = message.get("sample_id", "N/A")
            ground_truth = message.get("ground_truth", "N/A")
            amostra_array = np.array(json.loads(message["amostra"]), dtype=float)

            print(f"\n[{self.node_id.upper()}] (SampleReceiver) Amostra {sample_id} recebida de {addr[0]}.")

            decision = self.traffic_fn(amostra_array, ground_truth)

            response = {
                "node_id": self.node_id,
                "sample_id": sample_id,
                "decision": str(decision),
                "status": "OK",
            }
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

        except Exception as e:
            print(f"[{self.node_id.upper()}] (SampleReceiver) Erro processando amostra {sample_id}: {e}")
            decision = f"ERROR: {type(e).__name__}"
            try:
                conn.sendall((json.dumps({
                    "node_id": self.node_id, "sample_id": sample_id, "status": "ERROR"
                }) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()
            processing_time_ms = (time.time() - start_time) * 1000
            if hasattr(self.logger, "log_amostra_recebida"):
                self.logger.log_amostra_recebida(
                    node_id=self.node_id,
                    sample_id=sample_id,
                    ground_truth=ground_truth,
                    decisao=decision,
                    tempo_proc_ms=processing_time_ms,
                )

    def start_listening(self):
        self.is_running = True
        self.server_socket.listen(20)  # fila maior: tráfego pode chegar em rajadas
        print(f"[{self.node_id.upper()}] (SampleReceiver) Escutando amostras em {self.host}:{self.port}...")

        def listen_thread():
            while self.is_running:
                try:
                    conn, addr = self.server_socket.accept()
                    threading.Thread(target=self._handle_sample, args=(conn, addr), daemon=True).start()
                except OSError:
                    break
                except Exception as e:
                    if self.is_running:
                        print(f"[{self.node_id.upper()}] (SampleReceiver) Erro inesperado: {e}")
                    break

        threading.Thread(target=listen_thread, daemon=True).start()

    def stop_listening(self):
        self.is_running = False
        self.server_socket.close()