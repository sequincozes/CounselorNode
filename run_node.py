import time
import numpy as np
import argparse

# 1. Atualizado: Importando o GossipNode
from core.node import GossipNode  
from infrastructure.networking import detect_local_ip

def parse_args():
    parser = argparse.ArgumentParser(description="Gossip Network Node")

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
        help="Tempo (em segundos) antes do nó começar a atuar maliciosamente"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Detecta o IP local
    local_ip = detect_local_ip()
    print(f"IP local detectado: {local_ip}")
    print("Iniciando nó...")

    if args.poison_rate > 0:
        print(f"⚠ Nó MALICIOSO iniciado com taxa de envenenamento de {args.poison_rate*100}% após {args.delay} segundos")
    else:
        print("✅ Nó HONESTO iniciado (sem envenenamento)")

    try:
        # 2. Atualizado: Usando GossipNode e PASSANDO OS ARGUMENTOS CORRETAMENTE
        node = GossipNode(
            detected_ip=local_ip,
            local_port=args.port,
            poison_rate=args.poison_rate,
            delay=args.delay
        )
        
        print("Iniciando serviços de rede e fofoca (Gossip)...")
        node.start()
        print("Serviços iniciados com sucesso.")

        # Dá tempo para a configuração inicial e descoberta de vizinhos
        print("Aguardando 10s para inicialização do motor de ML e da rede...")
        time.sleep(10)

        print("\n" + "=" * 60)
        print(f"📡 SIMULAÇÃO: {node.node_id.upper()} LENDO TRÁFEGO DE REDE")
        print("=" * 60)

        # Loop de Simulação de Tráfego
        rodada = 1
        while True:
            # Pega a próxima amostra não processada (sequencial)
            suspect_sample_data, ground_truth = node.generate_next_sample()
            
            # Verifica se todas as amostras foram processadas
            if suspect_sample_data is None:
                print(f"\n[{node.node_id.upper()}] 🎯 SIMULAÇÃO CONCLUÍDA - Todas as amostras foram classificadas!")
                print(f"[{node.node_id.upper()}] Total de amostras processadas: {len(node.processed_samples)}")
                break
            
            # Passa a amostra e o ground truth para o nó avaliar
            # No Gossip, a decisão é instantânea pois o nó já aprendeu com os vizinhos
            result = node.check_traffic_and_act(suspect_sample_data, ground_truth)

            # Aqui você pode salvar logs adicionais se quiser, mas o node.py já loga a decisão.
            time.sleep(2)  # Aumentei o sleep para 2s para facilitar a leitura no terminal
            rodada += 1

    except KeyboardInterrupt:
        print("\n[INFO] Desligamento solicitado pelo usuário. Encerrando o nó...")
    except Exception as e:
        print(f"\n[ERRO FATAL] Ocorreu um erro inesperado: {e}")

if __name__ == '__main__':
    main()