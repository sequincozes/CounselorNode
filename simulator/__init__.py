import socket
import time
import multiprocessing as mp
import copy

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


def node_process(ip, port, ml_override=None):
    try:
        if ml_override:
            ML_OVERRIDES_BY_PORT[port] = ml_override

        node = CounselorNode(detected_ip=ip, local_port=port)
        print(f"{Colors.OKBLUE}[SISTEMA] Nó {node.node_id} Online em {ip}:{port}{Colors.ENDC}")

        node.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{Colors.FAIL}[ERRO] Falha crítica no nó {port}: {e}{Colors.ENDC}")


def _pick_source(engine: ClassifierEngine, sample_source: str):
    """
    Retorna (X_src, y_src, src_name) priorizando SEMPRE os arrays RAW quando disponíveis.
    Isso evita enviar amostras escaladas ao solicitar conselho (double scaling nos peers).
    """
    # helpers: pega RAW se existir, senão usa scaled
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
        # pode existir mesmo sem CSV B
        X_src, is_raw = raw_or_scaled("X_eval_raw", "X_eval")
        y_src = engine.y_eval
        src_name = "EVAL(CSV A)" + (" [RAW]" if is_raw else " [SCALED/FALLBACK]")
        return X_src, y_src, src_name

    # default: train
    X_src, is_raw = raw_or_scaled("X_train_raw", "X_train")
    y_src = engine.y_train
    src_name = "TRAIN(CSV A)" + (" [RAW]" if is_raw else " [SCALED/FALLBACK]")
    return X_src, y_src, src_name


def run_trigger(ip="127.0.0.1", port=5000, sample_index=0, ml_override=None, sample_source="final_test"):
    if ml_override:
        ML_OVERRIDES_BY_PORT[port] = ml_override

    peer_manager = ConfigManager(ip, local_port=port)
    node_id = peer_manager.node_id

    logger = CounselorLogger(node_id, use_log_folder=False)
    ml_config = peer_manager.get_ml_config()

    engine = ClassifierEngine(ml_config)
    client = CounselorClient(node_id, peer_manager, logger)

    # Escolha do conjunto para amostragem (prioriza RAW)
    X_src, y_src, src_name = _pick_source(engine, sample_source)

    # Proteção de índice
    sample_index = int(sample_index)
    if sample_index < 0 or sample_index >= len(X_src):
        sample_index = 0

    # >>> IMPORTANTÍSSIMO: sample deve ser RAW (quando disponível) <<<
    sample = X_src[sample_index]
    ground_truth = y_src[sample_index]

    print(f"\n{Colors.BOLD}[SORTEIO]{Colors.ENDC} Fonte={src_name} | Amostra #{sample_index} selecionada.")
    print(f"{Colors.BOLD}[SORTEIO]{Colors.ENDC} Ground Truth Real: {Colors.WARNING}{ground_truth}{Colors.ENDC}")

    print(f"\n{Colors.OKGREEN}[*] Nó {node_id} processando tráfego e verificando conflitos...{Colors.ENDC}")
    print(f"[{node_id.upper()}] (Ground Truth para esta amostra: {ground_truth})")

    # Engine espera RAW e escala internamente
    results = engine.classify_and_check_conflict(sample)

    classification = results['classification']
    conflict = results['conflict']
    decisions = results['decisions']
    cluster_id = results.get('cluster_id', 'N/A')

    print(f"[{node_id.upper()}] Resultado DCS Local (Cluster {cluster_id}): {classification}")
    print(f"[{node_id.upper()}] Decisões Locais (Classes): {decisions}")

    initial_chain = [f"{ip}:{port}"]

    if conflict:
        print(f"[{node_id.upper()}] Alerta: CONFLITO DETECTADO! Consultando a Counselors Network.")

        other_peers = peer_manager.get_other_peers()
        if not other_peers:
            print(f"[{node_id.upper()}] Ação: Conflito detectado, mas não há outros pares. Usando decisão local padrão.")
            best_model_class = engine.counseling_logic(sample)  # sample RAW
            return "INTRUSION" if best_model_class != "NORMAL" else "NORMAL"

        # >>> Envia RAW para os peers (cada peer aplica seu scaler local) <<<
        counsel_response = client.request_counsel(sample, other_peers, ground_truth, initial_chain)
        if counsel_response and counsel_response.get("decision") == "INTRUSION":
            return "INTRUSION"
        return "NORMAL"

    return "INTRUSION" if classification != "NORMAL" else "NORMAL"


def main():
    print(f"{Colors.HEADER}{Colors.BOLD}=== SIMULADOR DE REDE P2P COUNSELORS (MULTIPROCESS) ==={Colors.ENDC}")

    # Exemplo: CSV A (treino+avaliação) e CSV B (teste final)
    nodes_config = [
        ("127.0.0.1", 5000, {
            "train_eval_dataset_source": "dataset_01_400_train.csv",
            "final_test_dataset_source": "dataset_01_100_test.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80
        }),
        ("127.0.0.1", 5001, {
            "train_eval_dataset_source": "dataset500multiclass.csv",
            "final_test_dataset_source": "dataset500multiclass.csv",
            "target_column": "class",
            "eval_size": 0.30,
            "clustering_n_clusters": 5,
            "f1_threshold": 0.05,
            "f1_min_required": 0.80,
            "outlier_enabled": True
        }),
        ("127.0.0.1", 5002, {
            "train_eval_dataset_source": "dataset500multiclass.csv",
            "final_test_dataset_source": "dataset500multiclass.csv",
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
    for ip, port, ml_override in nodes_config:
        p = mp.Process(target=node_process, args=(ip, port, ml_override))
        p.start()
        procs.append(p)

    print(f"{Colors.WARNING}[*] Aguardando nós ficarem online (portas abertas)...{Colors.ENDC}")

    peer_ports = [(ip, port) for ip, port, _ in nodes_config]
    not_ready = wait_for_peer_ports(peer_ports, timeout_sec=180, interval_sec=1.0)

    if not_ready:
        print(f"{Colors.WARNING}[!] Atenção: alguns nós ainda não estão online: {not_ready}{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}[*] Todos os nós estão online.{Colors.ENDC}")

    ml_override_no1 = nodes_config[0][2]
    resultado = run_trigger(
        ip="127.0.0.1",
        port=5000,
        sample_index=1,
        ml_override=ml_override_no1,
        sample_source="final_test"  # final_test / eval / train
    )

    print(f"\n{Colors.HEADER}=== RESULTADO DA SIMULAÇÃO ==={Colors.ENDC}")
    print(f"Decisão Final do Sistema: {Colors.BOLD}{resultado}{Colors.ENDC}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}[!] Encerrando simulação...{Colors.ENDC}")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()

def wait_for_peer_ports(peers, timeout_sec=120, interval_sec=1.0):
    """
    Espera até que todas as portas dos peers estejam aceitando conexão TCP.
    peers: lista de tuplas (ip, port)
    """
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

    return list(pending)  # retorna as que não subiram

if __name__ == "__main__":
    main()
