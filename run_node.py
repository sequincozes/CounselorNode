import sys
import time
import numpy as np
from core.node import CounselorNode


# Função auxiliar para gerar amostras de teste consistentes
def generate_test_sample(engine):
    """Gera uma amostra aleatória do conjunto de teste do ClassifierEngine."""
    if engine.X_test is not None and len(engine.X_test) > 0:
        idx = np.random.randint(0, len(engine.X_test))
        # Retorna a amostra (já padronizada)
        return engine.X_test[idx]

    return np.zeros(engine.config['n_features'])


def main():
    if len(sys.argv) < 2:
        print("Uso: python run_node.py <porta>")
        print("Exemplo: python run_node.py 5000")
        sys.exit(1)

    try:
        local_port = int(sys.argv[1])
    except ValueError:
        print("Erro: A porta deve ser um número inteiro.")
        sys.exit(1)

    try:
        node = CounselorNode(local_port)
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
            suspect_sample_data = generate_test_sample(node.engine)
            node.check_traffic_and_act(suspect_sample_data)

            time.sleep(5)  # Espera entre as amostras


    except KeyboardInterrupt:
        print("\nPrograma terminado pelo usuário.")
    except Exception as e:
        print(f"Erro inesperado: {e}")


if __name__ == '__main__':
    main()