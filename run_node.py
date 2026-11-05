"""
run_node.py — Ponto de entrada para execução de um único CounselorNode em terminal.

Uso
---
    python run_node.py <PORTA> [opções]

Exemplos
--------
  Terminal 1 (nó 1, porta 5000):
      python run_node.py 5000

  Terminal 2 (nó 2, porta 5001):
      python run_node.py 5001

  Terminal 3 (nó 3, malicioso com 90% de envenenamento, porta 5002):
      python run_node.py 5002 --poison-rate 0.9

Parâmetros de datasets:
  --train-dataset    Caminho para o CSV de treino/validação (sobrescreve env e config)
  --test-dataset     Caminho para o CSV de teste final (sobrescreve env e config)

Variáveis de ambiente suportadas:
  $env:TRAIN_EVAL_DATASET_SOURCE = "data_sbrc2026/no2.csv"
  $env:FINAL_TEST_DATASET_SOURCE = "data_sbrc2026/no2.csv"

Parâmetros de defesa (desafios + reputação):
  --tau              Limiar de reputação para isolamento (padrão: 0.6)
  --min-challenges   Mínimo de desafios antes de isolar um par (padrão: 10)
"""

import time
import argparse
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.node import CounselorNode
from infrastructure.networking import detect_local_ip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Counselor Node — IDS colaborativo P2P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "port",
        type=int,
        help="Porta TCP onde o nó irá escutar pedidos de conselho"
    )

    # --- Datasets ---
    parser.add_argument(
        "--train-dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Caminho para dataset de treino (sobrescreve config e variáveis de ambiente)"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Caminho para dataset de teste final (sobrescreve config e variáveis de ambiente)"
    )

    # --- Envenenamento ---
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.0,
        metavar="RATE",
        help="Taxa de envenenamento de respostas (0.0 = honesto, 1.0 = sempre mente). Padrão: 0.0"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        metavar="SEGUNDOS",
        help="Tempo (s) antes do nó começar a envenenar conselhos. Padrão: 0"
    )

    # --- Mecanismo de defesa ---
    parser.add_argument(
        "--tau",
        type=float,
        default=0.6,
        metavar="TAU",
        help="Limiar de reputação mínima τ [0,1]. Par com rep < τ após min-challenges é isolado. Padrão: 0.6"
    )
    parser.add_argument(
        "--min-challenges",
        type=int,
        default=10,
        metavar="N",
        help="Nº mínimo de desafios para habilitar o isolamento de um par. Padrão: 10"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Se informado via CLI, sobrepõe variáveis de ambiente
    if args.train_dataset:
        os.environ["TRAIN_EVAL_DATASET_SOURCE"] = args.train_dataset
    if args.test_dataset:
        os.environ["FINAL_TEST_DATASET_SOURCE"] = args.test_dataset

    local_ip = detect_local_ip()
    print(f"[run_node] IP local detectado: {local_ip}")
    print(f"[run_node] Porta: {args.port}")
    if os.environ.get("TRAIN_EVAL_DATASET_SOURCE"):
        print(f"[run_node] Dataset Treino (ENV/CLI): {os.environ.get('TRAIN_EVAL_DATASET_SOURCE')}")
    if os.environ.get("FINAL_TEST_DATASET_SOURCE"):
        print(f"[run_node] Dataset Teste (ENV/CLI):  {os.environ.get('FINAL_TEST_DATASET_SOURCE')}")
    print(f"[run_node] Defesa — τ={args.tau} | min_challenges={args.min_challenges}")

    if args.poison_rate > 0:
        print(f"[run_node] ⚠ NÓ MALICIOSO: poison_rate={args.poison_rate*100:.0f}% | delay={args.delay}s")
    else:
        print("[run_node] Nó honesto (sem envenenamento)")

    try:
        node = CounselorNode(
            detected_ip=local_ip,
            local_port=args.port,
            poison_rate=args.poison_rate,
            delay=args.delay,
            defense_tau=args.tau,
            defense_min_challenges=args.min_challenges,
        )

        node.start()

        print(f"\n{'='*60}")
        print(f"  NÓ {node.node_id.upper()} ONLINE")
        print(f"  Porta de conselhos : {args.port}")
        print(f"  Porta de amostras  : {node.sample_port}")
        print(f"  Reputação — τ={args.tau} | min_challenges={args.min_challenges}")
        print(f"{'='*60}")
        print("  Aguardando amostras via SampleReceiver...")
        print("  (Ctrl+C para encerrar)\n")

        # Mantém o processo vivo; o trabalho real acontece nas threads de servidor
        while True:
            time.sleep(60)
            # Exibe snapshot periódico do estado de reputação
            estados = node.reputation_manager.get_all_states()
            if estados:
                print(f"\n[{node.node_id.upper()}] --- Snapshot de Reputação ---")
                for peer_key, info in estados.items():
                    flag = "⚠ ISOLATED" if info['estado'] == 'ISOLATED' else info['estado']
                    print(f"  {info['name']:8s} ({peer_key}) | "
                          f"rep={info['rep']:.4f} | s={info['s']} f={info['f']} | {flag}")
                print()

    except KeyboardInterrupt:
        print(f"\n[{node.node_id.upper()}] Encerrando nó...")

        # Exibe estado final de reputação
        print(f"\n--- Estado Final de Reputação ({node.node_id.upper()}) ---")
        for peer_key, info in node.reputation_manager.get_all_states().items():
            flag = "⚠ ISOLATED" if info['estado'] == 'ISOLATED' else info['estado']
            print(f"  {info['name']:8s} | rep={info['rep']:.4f} | "
                  f"s={info['s']} f={info['f']} | n={info['n_chall']} | {flag}")

        # Snapshot final de métricas
        node.reputation_manager.snapshot_metricas_defesa()
        print("Logs salvos. Até logo.")

    except Exception as e:
        import traceback
        print(f"[run_node] ERRO: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
