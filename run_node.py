import sys
import time
import numpy as np
from core.node import CounselorNode
from infrastructure.networking import detect_local_ip  # Importa a função de detecção


# Função auxiliar para gerar amostras de teste consistentes
def generate_test_sample(engine):
    """
    Gera uma amostra aleatória do conjunto de teste do ClassifierEngine
    e seu respectivo ground truth (rótulo real).
    """
    if engine.X_test is not None and len(engine.X_test) > 0:
        idx = np.random.randint(0, len(engine.X_test))
        # Retorna a amostra (já padronizada) e o rótulo real (ground truth)
        return engine.X_test[idx], engine.y_test[idx]

    # Fallback
    return np.zeros(engine.config['n_features']), "N/A"


def main():
    # 1. Detecta o IP local
    local_ip = detect_local_ip()
    print(f"IP local detectado: {local_ip}")
    print("Iniciando nó conselheiro...")

    try:
        # 2. Inicializa o nó usando o IP detectado como identificador
        node = CounselorNode(local_ip)
        node.start()

        # Dá tempo para a configuração (treinamento de ML pode levar alguns segundos)
        # e para outros nós iniciarem.
        print("Aguardando 10s para inicialização do motor de ML e da rede...")
        time.sleep(10)

        print("\n" + "=" * 50)
        print(f"SIMULAÇÃO: {node.node_id.upper()} (DCS) VERIFICANDO TRÁFEGO")
        print("=" * 50)

        # Simulação de Tráfego (agora todos os nós fazem isso)
        while True:
            # Amostra 1: Deve ser classificada ou gerar conflito
            # suspect_sample_data, ground_truth = generate_test_sample(node.engine)

            # Passa a amostra e o ground truth para o nó
            # node.check_traffic_and_act(suspect_sample_data, ground_truth)

            time.sleep(5)  # Espera entre as amostras


    except KeyboardInterrupt:
        print("\nPrograma terminado pelo usuário.")
    except Exception as e:
        print(f"Erro inesperado: {e}")


if __name__ == '__main__':
    main()