import time
import multiprocessing as mp

from core.node import CounselorNode
from infrastructure.config_manager import ConfigManager
from infrastructure.logger import CounselorLogger
from infrastructure.networking import CounselorClient
from core.classifier_engine import ClassifierEngine


# Cores ANSI para o terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ------------------------------------------------------------
# 1) HOTFIX (somente no simulador): get_other_peers por IP+PORTA
# ------------------------------------------------------------
def _get_other_peers_ip_port(self):
    return [
        p for p in self.peers
        if not (p['ip'] == self.local_ip and p['port'] == self.local_port)
    ]


ConfigManager.get_other_peers = _get_other_peers_ip_port


# ------------------------------------------------------------
# 2) Worker de processo: sobe um CounselorNode (servidor)
# ------------------------------------------------------------
def node_process(ip, port):
    try:
        node = CounselorNode(detected_ip=ip, local_port=port)
        print(f"{Colors.OKBLUE}[SISTEMA] Nó {node.node_id} Online em {ip}:{port}{Colors.ENDC}")

        node.start()  # ATENÇÃO: no seu projeto isso NÃO bloqueia

        # Mantém o processo vivo; caso contrário, o servidor (thread daemon) morre e a porta cai.
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{Colors.FAIL}[ERRO] Falha crítica no nó {port}: {e}{Colors.ENDC}")


# ------------------------------------------------------------
# 3) Trigger sem bind: não cria CounselorNode (evita conflito)
# ------------------------------------------------------------
def run_trigger(ip="127.0.0.1", port=5000, sample_index=1463):
    peer_manager = ConfigManager(ip, local_port=port)
    node_id = peer_manager.node_id

    logger = CounselorLogger(node_id, use_log_folder=False)
    ml_config = peer_manager.get_ml_config()

    engine = ClassifierEngine(ml_config)
    client = CounselorClient(node_id, peer_manager, logger)

    indice_sorteado = sample_index
    sample = engine.X_train[indice_sorteado]
    ground_truth = engine.y_train[indice_sorteado]

    print(f"\n{Colors.BOLD}[SORTEIO]{Colors.ENDC} Amostra #{indice_sorteado} selecionada.")
    print(f"{Colors.BOLD}[SORTEIO]{Colors.ENDC} Ground Truth Real: {Colors.WARNING}{ground_truth}{Colors.ENDC}")

    print(f"\n{Colors.OKGREEN}[*] Nó {node_id} processando tráfego e verificando conflitos...{Colors.ENDC}")
    print(f"[{node_id.upper()}] (Ground Truth para esta amostra: {ground_truth})")

    results = engine.classify_and_check_conflict(sample)

    classification = results['classification']
    conflict = results['conflict']
    decisions = results['decisions']
    cluster_id = results.get('cluster_id', 'N/A')

    print(f"[{node_id.upper()}] Resultado DCS Local (Cluster {cluster_id}): {classification}")
    print(f"[{node_id.upper()}] Decisões Locais (Classes): {decisions}")

    initial_chain = [f"{ip}:{port}"]

    if conflict:
        print(f"[{node_id.upper()}] Alerta: CONFLITO DE CLASSIFICADOR DETECTADO! Consultando a Counselors Network.")

        other_peers = peer_manager.get_other_peers()
        if not other_peers:
            print(f"[{node_id.upper()}] Ação: Conflito detectado, mas não há outros pares. Usando decisão local padrão.")
            best_model_class = engine.counseling_logic(sample)
            return "INTRUSION" if best_model_class != "NORMAL" else "NORMAL"

        counsel_response = client.request_counsel(sample, other_peers, ground_truth, initial_chain)

        if counsel_response and counsel_response.get("decision") == "INTRUSION":
            final_decision = "INTRUSION"
        else:
            final_decision = "NORMAL"

        return final_decision

    return "INTRUSION" if classification != "NORMAL" else "NORMAL"


def main():
    print(f"{Colors.HEADER}{Colors.BOLD}=== SIMULADOR DE REDE P2P COUNSELORS (MULTIPROCESS) ==={Colors.ENDC}")

    nodes_config = [("127.0.0.1", 5000), ("127.0.0.1", 5001), ("127.0.0.1", 5002)]

    mp.set_start_method("spawn", force=True)

    procs = []
    for ip, port in nodes_config:
        # IMPORTANTE: não use daemon=True aqui
        p = mp.Process(target=node_process, args=(ip, port))
        p.start()
        procs.append(p)

    print(f"{Colors.WARNING}[*] Aguardando 5s para inicialização dos motores de ML...{Colors.ENDC}")
    time.sleep(5)

    resultado = run_trigger(ip="127.0.0.1", port=5000, sample_index=1463)

    print(f"\n{Colors.HEADER}=== RESULTADO DA SIMULAÇÃO ==={Colors.ENDC}")
    print(f"Decisão Final do Sistema: {Colors.BOLD}{resultado}{Colors.ENDC}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}[!] Encerrando simulação...{Colors.ENDC}")
    finally:
        # Encerra processos dos nós ao sair
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()
