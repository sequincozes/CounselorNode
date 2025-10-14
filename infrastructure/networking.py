import json
import socket
import threading
import numpy as np

BUFFER_SIZE = 1024


class CounselorServer:
    """Implements the server that listens for P2P counseling requests."""

    def __init__(self, host, port, node_id, local_source_type, counseling_fn):
        self.host = host
        self.port = port
        self.node_id = node_id
        self.local_source_type = local_source_type
        self.counseling_logic_fn = counseling_fn  # Callback function for ML logic
        self.is_running = False

    def _handle_request(self, conn, addr):
        """Processes a received counseling request."""
        try:
            data = conn.recv(BUFFER_SIZE).decode('utf-8')
            if not data: return

            request = json.loads(data)
            requester_id = request.get('requester_id', 'Unknown')
            amostra_str = request.get('amostra', 'N/A')

            # --- REAL CLASSIFICATION LOGIC ON THE SERVER ---
            try:
                # Converts the JSON feature string back to a NumPy array
                amostra_array = np.array(json.loads(amostra_str), dtype=float)

                # Executes the real classification logic (high confidence)
                final_prediction_str = self.counseling_logic_fn(amostra_array)

                # Formats the response
                if final_prediction_str == '0':  # Example: assuming class '0' is NORMAL
                    decision = "NORMAL"
                    counsel_msg = "Analysis concluded. Result: Normal traffic (High Confidence - DCS)."
                elif final_prediction_str == 'UNKNOWN':
                    decision = "UNKNOWN"
                    counsel_msg = "Analysis failed: Cluster not mapped on Counselor Node."
                else:  # Any other class is INTRUSION
                    decision = "INTRUSION"
                    counsel_msg = f"Intrusion Confirmed: Class {final_prediction_str} (High Confidence via DCS)."

            except Exception as e:
                print(f"SERVER ERROR: Classification failed: {e}")
                decision = "ERROR"
                counsel_msg = "Classification engine failed on this sample."
            # ------------------------------------------------

            print(f"\n[SERVER] Received counseling request from {requester_id} ({addr[0]}:{addr[1]})")

            response = {
                "counselor_id": self.node_id,
                "decision": decision,
                "counsel": counsel_msg,
                "source_type": self.local_source_type
            }

            conn.sendall(json.dumps(response).encode('utf-8'))
            print(f"[SERVER] Counsel sent: '{response['decision']}'.")

        except Exception as e:
            print(f"[SERVER] Error processing request: {e}")
        finally:
            conn.close()

    def start_listening(self):
        """Starts the server in a separate thread to listen for P2P requests."""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.is_running = True

            print(f"[SERVER] Listening for connections on {self.host}:{self.port}...")

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
                            print(f"[SERVER] Error in main loop: {e}")
                        break

            server_thread = threading.Thread(target=server_loop)
            server_thread.daemon = True
            server_thread.start()

        except Exception as e:
            print(f"[SERVER] Error starting server: {e}")
            raise


class CounselorClient:
    """Implements the client functionality to request counsel from other peers."""

    def __init__(self, node_id, peer_manager):
        self.node_id = node_id
        self.peer_manager = peer_manager

    def request_counsel(self, sample_data_array, required_source_type):
        """
        Selects a peer and sends a counseling request.
        Accepts sample_data as a NumPy array.
        """
        print("\n--- REQUESTING P2P COUNSEL ---")

        target_peer = self.peer_manager.find_counselor(required_source_type)

        if not target_peer:
            print(f"[CLIENT] Counselor with specialty '{required_source_type}' not found.")
            return None

        peer_ip = target_peer['ip']
        peer_port = target_peer['port']

        print(f"[CLIENT] Selected Counselor: {target_peer['name']} ({peer_ip}:{peer_port})")

        # Converts the NumPy array (sample_data_array) to a serializable JSON string
        sample_data_str = json.dumps(sample_data_array.tolist())

        request_data = {
            "requester_id": self.node_id,
            "reason": "Local classifier conflict",
            "amostra": sample_data_str  # Sent as a JSON array string
        }
        request_message = json.dumps(request_data).encode('utf-8')

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((peer_ip, peer_port))
            client_socket.sendall(request_message)

            response_data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
            response = json.loads(response_data)

            print("--- COUNSEL RECEIVED ---")
            print(f"Counselor Decision ({response['counselor_id']}): {response['decision']}")
            print("--------------------------")
            return response

        except socket.timeout:
            print(f"[CLIENT] Error: Timeout attempting to communicate with {peer_ip}:{peer_port}")
        except ConnectionRefusedError:
            print(f"[CLIENT] Error: Connection refused. Counselor is not active.")
        except Exception as e:
            print(f"[CLIENT] Error in P2P communication: {e}")
        finally:
            if 'client_socket' in locals():
                client_socket.close()

        return None