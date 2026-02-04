import os
import socket
import sys
import time
import multiprocessing as mp
import copy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from core.node import CounselorNode
from infrastructure.config_manager import ConfigManager
from infrastructure.logger import CounselorLogger
from infrastructure.networking import CounselorClient
from core.classifier_engine import ClassifierEngine

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def wait_for_peer_ports(peers, timeout_sec=120, interval_sec=1.0):
    deadline = time.time() + timeout_sec
    pending = set(peers)

    while pending and time.time() < deadline:
        to_remove = set()
        for ip, port in pending:
            try:
                with socket.create_connection((ip, port), timeout=0.5):
                    to_remove.add((ip, port))
            except OSError:
                pass

        pending -= to_remove
        if pending:
            time.sleep(interval_sec)

    return list(pending)


def _get_other_peers_ip_port(self):
    return [
        p for p in self.peers
        if not (p['ip'] == self.local_ip and p['port'] == self.local_port)
    ]


def _get_ml_config_with_override(self):
    cfg = copy.deepcopy(self.ml_config)
    ov = ML_OVERRIDES_BY_PORT.get(self.local_port)
    if ov:
        cfg.update(ov)
    return cfg


ML_OVERRIDES_BY_PORT = {}

ConfigManager.get_ml_config = _get_ml_config_with_override
ConfigManager.get_other_peers = _get_other_peers_ip_port


def node_process(ip, port, ml_override=None, poison_rate=1, delay=0):
    try:
        if ml_override:
            ML_OVERRIDES_BY_PORT[port] = ml_override

        node = CounselorNode(detected_ip=ip, local_port=port, poison_rate=poison_rate, delay=delay)
        print(f"{Colors.OKBLUE}[SISTEMA] Nó {node.node_id} Online em {ip}:{port}{Colors.ENDC}")

        node.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{Colors.FAIL}[ERRO] Falha crítica no nó {port}: {e}{Colors.ENDC}")


def _pick_source(engine: ClassifierEngine, sample_source: str):
    def raw_or_scaled(raw_attr, scaled_attr):
        if hasattr(engine, raw_attr) and getattr(engine, raw_attr) is not None:
            return getattr(engine, raw_attr), True
        return getattr(engine, scaled_attr), False

    if sample_source == "final_test" and getattr(engine, "X_final_test", None) is not None:
        X_src, is_raw = raw_or_scaled("X_final_test_raw", "X_final_test")
        y_src = engine.y_final_test
        src_name = "FINAL_TEST(CSV B)" + (" [RAW]" if is_raw else " [SCALED/FALLBACK]")
        return X_src, y_src, src_name

    if sample_source == "eval":
        X_src, is_raw = raw_or_scaled("X_eval_raw", "X_eval")
        y_src = engine.y_eval
        src_name = "EVAL(CSV A)" + (" [RAW]" if is_raw else " [SCALED/FALLBACK]")
        return X_src, y_src, src_name

    X_src, is_raw = raw_or_scaled("X_train_raw", "X_train")
    y_src = engine.y_train
    src_name = "TRAIN(CSV A)" + (" [RAW]" if is_raw else " [SCALED/FALLBACK]")
    return X_src, y_src, src_name


def run_trigger_with_instances(peer_manager, node_id, engine, client, sample_index=0, sample_source="final_test"):
    X_src, y_src, src_name = _pick_source(engine, sample_source)

    sample_index = int(sample_index)
    if sample_index < 0 or sample_index >= len(X_src):
        sample_index = 0

    sample_raw = X_src[sample_index]
    ground_truth = y_src[sample_index]

    print(f"\n{Colors.BOLD}[SORTEIO]{Colors.ENDC} Fonte={src_name} | Amostra #{sample_index} selecionada.")
    print(f"{Colors.BOLD}[SORTEIO]{Colors.ENDC} Ground Truth Real: {Colors.WARNING}{ground_truth}{Colors.ENDC}")

    print(f"\n{Colors.OKGREEN}[*] Nó {node_id} processando tráfego e verificando conflitos...{Colors.ENDC}")
    print(f"[{node_id.upper()}] (Ground Truth para esta amostra: {ground_truth})")

    results = engine.classify_and_check_conflict(sample_raw)

    classification = results['classification']
    conflict = results['conflict']
    decisions = results['decisions']
    cluster_id = results.get('cluster_id', 'N/A')

    print(f"[{node_id.upper()}] Resultado DCS Local (Cluster {cluster_id}): {classification}")
    print(f"[{node_id.upper()}] Decisões Locais (Classes): {decisions}")

    initial_chain = [f"{peer_manager.local_ip}:{peer_manager.local_port}"]

    if conflict:
        print(f"[{node_id.upper()}] Alerta: CONFLITO DETECTADO! Consultando a Counselors Network.")

        other_peers = peer_manager.get_other_peers()
        if not other_peers:
            print(f"[{node_id.upper()}] Ação: Conflito detectado, mas não há outros pares. Usando decisão local padrão.")
            best_model_class = engine.counseling_logic(sample_raw)
            return best_model_class, sample_index, sample_raw, ground_truth, conflict

        counsel_response = client.request_counsel(sample_raw, other_peers, ground_truth, initial_chain)
        return counsel_response, sample_index, sample_raw, ground_truth, conflict

    return classification, sample_index, sample_raw, ground_truth, conflict


def main():
    import os
    print("[DEBUG] CWD =", os.getcwd())

    print(f"{Colors.HEADER}{Colors.BOLD}=== SIMULADOR DE REDE P2P COUNSELORS (MULTIPROCESS) ==={Colors.ENDC}")

    nodes_config = [
        ("127.0.0.1", 5000, {
            "train_eval_dataset_source": "datasets/sbrc/treino_no1.csv",
            "final_test_dataset_source": "datasets/sbrc/zero_day.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80
        }),
        ("127.0.0.1", 5001, {
            "train_eval_dataset_source": "datasets/sbrc/no2.csv",
            "final_test_dataset_source": "datasets/sbrc/no2.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
        ("127.0.0.1", 5002, {
            "train_eval_dataset_source": "datasets/sbrc/no3.csv",
            "final_test_dataset_source": "datasets/sbrc/no3.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
    ]

    mp.set_start_method("spawn", force=True)

    procs = []
    poison_rate = 1
    delay = 0.0
    for ip, port, ml_override in nodes_config:
        p = mp.Process(target=node_process, args=(ip, port, ml_override, poison_rate, delay))
        p.start()
        procs.append(p)

    print(f"{Colors.WARNING}[*] Aguardando nós ficarem online (portas abertas)...{Colors.ENDC}")
    peer_ports = [(ip, port) for ip, port, _ in nodes_config]
    not_ready = wait_for_peer_ports(peer_ports, timeout_sec=180, interval_sec=1.0)

    if not_ready:
        print(f"{Colors.WARNING}[!] Atenção: alguns nós ainda não estão online: {not_ready}{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}[*] Todos os nós estão online.{Colors.ENDC}")

    # Instâncias UMA vez (para manter aprendizado no loop)
    ml_override_no1 = nodes_config[0][2]
    ML_OVERRIDES_BY_PORT[5000] = ml_override_no1

    peer_manager = ConfigManager("127.0.0.1", local_port=5000)
    node_id = peer_manager.node_id

    logger = CounselorLogger(node_id, use_log_folder=True)
    ml_config = peer_manager.get_ml_config()

    engine = ClassifierEngine(ml_config)
    client = CounselorClient(node_id, peer_manager, logger)

    sample_source = "final_test"
    sample_index = 0
    X_src, _, _ = _pick_source(engine, sample_source)
    max_samples = len(X_src)

    try:
        while sample_index < max_samples:
            resultado, used_index, sample_raw, ground_truth, conflict = run_trigger_with_instances(
                peer_manager=peer_manager,
                node_id=node_id,
                engine=engine,
                client=client,
                sample_index=sample_index,
                sample_source=sample_source,
            )

            print(f"\n{Colors.HEADER}=== RESULTADO DA AMOSTRA ==={Colors.ENDC}")

            # Se resultado é dict, veio conselho
            if isinstance(resultado, dict):
                learned_label = resultado.get("decision", None)
                counselor_id = (
                    resultado.get("name_conselheiro")
                    or resultado.get("from_counselor")
                    or resultado.get("counselor_id")
                    or "UNKNOWN"
                )

                print(f"Amostra #{used_index} | Conselho: {learned_label} (de {counselor_id})")

                # >>> NOVO LOG (decisões) <<<
                decisao_final = learned_label if learned_label is not None else "UNKNOWN"
                responsavel = counselor_id
                logger.log_decisao(ground_truth, decisao_final, responsavel)

                if learned_label is not None and learned_label not in ["UNKNOWN", "LOOP_CLOSED"]:
                    # Aprende
                    engine.add_training_sample_raw(sample_raw, learned_label, retrain=False)
                    engine.rebuild()

                    # Log snapshot (com F1 por classificador)
                    long_rows = engine.get_long_f1_rows(
                        rodada=used_index,
                        sample_raw=sample_raw,
                        decisao=learned_label,
                        conflito=conflict
                    )
                    logger.log_cluster_classifier_f1_long(long_rows)

                    print(f"{Colors.OKGREEN}[LEARN]{Colors.ENDC} Aprendeu label={learned_label} e atualizou o log.")
                else:
                    long_rows = engine.get_long_f1_rows(
                        rodada=used_index,
                        sample_raw=sample_raw,
                        decisao=str(resultado),
                        conflito=conflict
                    )
                    logger.log_cluster_classifier_f1_long(long_rows)
                    print(f"{Colors.WARNING}[LEARN]{Colors.ENDC} Sem label útil para aprendizado (decision={learned_label}).")

            else:
                # Sem conselho
                print(f"Amostra #{used_index} | Decisão Local: {Colors.BOLD}{resultado}{Colors.ENDC}")

                decisao_final = str(resultado)
                responsavel = "local"
                logger.log_decisao(ground_truth, decisao_final, responsavel)

                long_rows = engine.get_long_f1_rows(
                    rodada=used_index,
                    sample_raw=sample_raw,
                    decisao=str(resultado),
                    conflito=conflict
                )
                logger.log_cluster_classifier_f1_long(long_rows)

            # Próxima amostra
            X_src, _, _ = _pick_source(engine, sample_source)
            sample_index = (used_index + 1) % len(X_src)

            # time.sleep(5)

    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}[!] Encerrando simulação...{Colors.ENDC}")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
