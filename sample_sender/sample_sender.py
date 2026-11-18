import json
import os
import socket
import csv
import argparse
import time
from datetime import datetime

import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sender_config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TargetNode:
    """Apenas o endereço de um nó de destino — sem dataset associado."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.ip = spec["ip"]
        self.port = spec["sample_port"]


class SampleDataset:
    """Fonte única de amostras, carregada uma vez e depois repartida entre os nós no envio."""

    def __init__(self, dataset_source, target_column=None):
        df = pd.read_csv(dataset_source)
        target_col = target_column or df.columns[-1]
        if target_col not in df.columns:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada em {dataset_source}")

        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        self.samples = X.tolist()
        self.ground_truths = y.tolist()
        self.cursor = 0  # cursor GLOBAL, único para todo o dataset

        n_features = len(self.samples[0]) if self.samples else 0
        print(f"[Dataset] Carregado: {len(self.samples)} amostras, {n_features} features (fonte única).")

    def has_next(self):
        return self.cursor < len(self.samples)

    def next_sample(self):
        """Avança o cursor global. Cada chamada consome UMA linha do dataset,
        não importa para qual nó ela será destinada em seguida."""
        idx = self.cursor
        features = self.samples[idx]
        ground_truth = self.ground_truths[idx]
        self.cursor += 1
        return idx, features, ground_truth


class AlternatingDispatch:
    """Alterna entre os nós em round-robin: 1ª amostra -> nó 1, 2ª -> nó 2, 3ª -> nó 3, 4ª -> nó 1..."""

    def __init__(self, targets):
        self.targets = targets
        self._index = 0

    def select_target(self):
        target = self.targets[self._index]
        self._index = (self._index + 1) % len(self.targets)
        return target


# TODO (modos futuros, mesma interface select_target()):
# class RandomDispatch: escolhe um nó aleatório a cada disparo.
# class ClassPercentageDispatch: direciona por percentual de classe a um nó específico.

DISPATCH_STRATEGIES = {
    "alternado": AlternatingDispatch,
}


class SampleSender:
    def __init__(self, config):
        self.dataset = SampleDataset(
            config["dataset_source"],
            target_column=config.get("target_column")
        )
        self.targets = [TargetNode(spec) for spec in config["targets"]]

        mode = config.get("dispatch_mode", "alternado")
        if mode not in DISPATCH_STRATEGIES:
            raise ValueError(f"Modo de disparo '{mode}' não implementado ainda.")
        self.dispatch_strategy = DISPATCH_STRATEGIES[mode](self.targets)

        # Timeout do socket (aumentado para permitir tempo de consulta P2P/retreino do nó)
        self.socket_timeout = config.get("socket_timeout_seconds", 60.0)

        # Intervalo padrão para o modo automático
        self.auto_interval = config.get("auto_interval_seconds", 30.0)

        self.report_path = config.get("report_output", "logs/sample_sender_report.csv")
        report_dir = os.path.dirname(self.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        self._init_report()

    def _init_report(self):
        is_new = not os.path.exists(self.report_path)
        self.report_file = open(self.report_path, "a", newline="", encoding="utf-8")
        self.report_writer = csv.writer(self.report_file)
        if is_new:
            self.report_writer.writerow(
                ["timestamp_envio", "node_destino", "sample_id", "classe_real", "status", "decisao_recebida"]
            )

    def _deliver(self, target: TargetNode, sample_id, features, ground_truth):
        message = {
            "sample_id": sample_id,
            "amostra": json.dumps(features),
            "ground_truth": str(ground_truth),
        }
        timestamp = datetime.now().isoformat()
        status, decision = "ERROR", "N/A"

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.socket_timeout)
                s.connect((target.ip, target.port))
                s.sendall((json.dumps(message) + "\n").encode("utf-8"))

                buffer = b""
                while b"\n" not in buffer:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                raw_response = buffer.partition(b"\n")[0].decode("utf-8")
                response = json.loads(raw_response) if raw_response else {}
                status = response.get("status", "UNKNOWN")
                decision = response.get("decision", "N/A")

            print(f"[{target.name}] Amostra {sample_id} enviada | "
                  f"ground_truth={ground_truth} | decisão={decision} | status={status}")

        except Exception as e:
            print(f"[{target.name}] ERRO ao enviar amostra {sample_id}: {e}")

        self.report_writer.writerow([timestamp, target.name, sample_id, ground_truth, status, decision])
        self.report_file.flush()

    def fire_next(self):
        """Puxa a PRÓXIMA linha do dataset único e entrega ao nó da vez (round-robin).
        Retorna True se enviou amostra, False se o dataset terminou."""
        if not self.dataset.has_next():
            print("[Dataset] Fim do fluxo — nenhuma amostra restante para enviar.")
            return False

        target = self.dispatch_strategy.select_target()
        sample_id, features, ground_truth = self.dataset.next_sample()
        self._deliver(target, sample_id, features, ground_truth)
        return True

    def run_manual_mode(self):
        print("\n=== Sample Sender — Modo Manual (disparo alternado) ===")
        print("Pressione ENTER para enviar a próxima amostra ao nó da vez. Digite 'q' + ENTER para sair.\n")

        while True:
            cmd = input("> ")
            if cmd.strip().lower() == "q":
                break
            if not self.dataset.has_next():
                print("[Dataset] Todas as amostras já foram enviadas.")
                break
            self.fire_next()

    def run_auto_mode(self, interval_seconds=None):
        interval = interval_seconds if interval_seconds is not None else self.auto_interval
        print(f"\n=== Sample Sender — Modo Automático (disparo alternado a cada {interval}s) ===")
        print("Pressione Ctrl+C para pausar/encerrar o envio automático.\n")

        sent_count = 0
        try:
            while self.dataset.has_next():
                sent_count += 1
                sent = self.fire_next()
                if not sent:
                    break

                if self.dataset.has_next():
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Próximo disparo em {interval}s...")
                    time.sleep(interval)

            if not self.dataset.has_next():
                print(f"\n[Modo Automático Finalizado] Todas as amostras foram enviadas! Total: {sent_count}")

        except KeyboardInterrupt:
            print(f"\n[Modo Automático Interrompido pelo usuário] Total de amostras enviadas nesta sessão: {sent_count}")

    def close(self):
        self.report_file.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample Sender — dispara amostras de um fluxo único, alternando entre nós"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="manual",
        choices=["manual", "auto"],
        help="Modo de operação: 'manual' (pressione ENTER a cada envio) ou 'auto' (disparo periódico contínuo). Padrão: manual."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="SEGUNDOS",
        help="Intervalo em segundos entre disparos no modo automático (sobrescreve config). Padrão: 30."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEGUNDOS",
        help="Tempo limite (s) de espera pela resposta do nó (sobrescreve config). Padrão: 60."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    if args.interval is not None:
        config["auto_interval_seconds"] = args.interval
    if args.timeout is not None:
        config["socket_timeout_seconds"] = args.timeout

    sender = SampleSender(config)
    try:
        if args.mode == "manual":
            sender.run_manual_mode()
        elif args.mode == "auto":
            interval = config.get("auto_interval_seconds", 30.0)
            sender.run_auto_mode(interval_seconds=interval)
    finally:
        sender.close()


if __name__ == "__main__":
    main()