import socket
import json

# Ajuste conforme necessário
HOST = "127.0.0.1"
PORT = 6000  # porta do sample_port do nó

# 25 features, conforme "n_features": 25 no peer_config.json
# Usando valores aleatórios só para verificar se a escuta responde;
# não espere uma classificação "correta" com dados fake.
amostra = [0.1] * 25

msg = {
    "sample_id": 1,
    "amostra": json.dumps(amostra),
    "ground_truth": "teste_manual"
}

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.sendall((json.dumps(msg) + "\n").encode("utf-8"))

resposta = s.recv(4096).decode("utf-8")
print("Resposta do nó:", resposta)
s.close()