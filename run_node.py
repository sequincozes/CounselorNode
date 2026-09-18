import time
import numpy as np
from core.node import CounselorNode
from infrastructure.networking import detect_local_ip
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Counselor Node")

    parser.add_argument(
        "port",
        type=int,
        help="Porta onde o nó irá escutar"
    )

    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.0,
        help="Taxa de envenenamento (0.0 a 1.0)"
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Tempo (em segundos) antes do nó começar a envenenar conselhos"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Detecta o IP local
    local_ip = detect_local_ip()
    print(f"IP local detectado: {local_ip}")
    print("Iniciando nó conselheiro...")

    if args.poison_rate > 0:
        print(f"Nó malicioso iniciado com taxa de envenenamento de {args.poison_rate*100}% após {args.delay} segundos")
    else:
        print("Nó honesto iniciado (sem envenenamento)")

    try:
        # 2. Inicializa o nó, agora passando a porta explicitamente
        node = CounselorNode(
            local_ip,
            local_port=args.port,
            poison_rate=args.poison_rate,
            delay=args.delay
        )

        print("Vai chamar o node.start...")
        node.start()
        print("O nó startou...")

        print("Aguardando 10s para inicialização do motor de ML e da rede...")
        time.sleep(10)

        print("\n" + "=" * 50)
        print(f"SIMULAÇÃO: {node.node_id.upper()} (DCS) VERIFICANDO TRÁFEGO")
        print("=" * 50)

        print("Nó pronto. Aguardando amostras via SampleReceiver...")
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nPrograma terminado pelo usuário.")
    except Exception as e:
        print(f"Erro inesperado: {e}")


if __name__ == '__main__':
    main()