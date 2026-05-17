import sys
import time
import numpy as np
import json
import threading
import random

# Importando camadas (Atualizado para refletir o novo modelo Gossip)
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import GossipServer, GossipClient
from infrastructure.logger import CounselorLogger
from core.classifier_engine import ClassifierEngine

class GossipNode:
    """A classe principal que representa o IDS (Detector) na rede de Gossip Learning."""

    # --- FILTRO DE LABELS INVÁLIDAS ---
    INVALID_LABELS = {
        "ERROR_CLASSIFICATION",
        "ERROR",
        "UNKNOWN",
        None
    }

    def __init__(self, detected_ip, local_port=None, poison_rate=1, delay=0):
        # 1. Configuração (Inicializado com o IP detectado)
        self.peer_manager = ConfigManager(detected_ip, local_port=local_port)
        local_info = self.peer_manager.get_local_info()

        # 2. Informações do nó
        self.node_id = self.peer_manager.node_id
        self.port = self.peer_manager.local_port
        self.local_ip = detected_ip
        self.config_ip = local_info.get('ip')
        self.bind_host = '0.0.0.0'

        # 3. Logger
        self.logger = CounselorLogger(self.node_id, use_log_folder=True)
        print(f"[{self.node_id.upper()}] Logger inicializado. Logs salvos na pasta padrão.")

        # 4. Configurações de Envenenamento (Malicious Node)
        self.poison_rate = poison_rate
        self.delay = delay
        self.start_time = time.time()

        # 5. Motor ML (Machine Learning)
        ml_config = self.peer_manager.get_ml_config()
        self.engine = ClassifierEngine(ml_config)
        # Controla se a ordem de processamento das amostras de teste deve ser aleatorizada
        self.randomize_test_order = bool(ml_config.get('randomize_test_order', False))
        seed = ml_config.get('test_random_seed', None)
        self._rand = random.Random(seed)
        # Controla se o nó deve fazer aprendizado online com cada amostra classificada
        self.online_learning_enabled = bool(ml_config.get('online_learning_enabled', True))

        # 6. Módulos de Rede Gossip
        self.client = GossipClient(self.node_id, self.peer_manager, self.logger)
        self.server = GossipServer(
            self.bind_host,
            self.port,
            self.node_id,
            self._execute_gossip_logic, # Callback para quando receber fofoca
            self.logger,
            self.peer_manager
        )

        # 7. Configurações do Loop de Gossip
        self.gossip_active = True
        self.gossip_interval = 4 # Frequência de fofoca (em segundos)
        self.gossip_thread = threading.Thread(target=self._gossip_loop, daemon=True)

        # 8. Controle de Amostras Processadas (para parada automática)
        self.processed_samples = set()  # Índices das amostras já processadas
        self.all_samples_processed = False  # Flag para indicar quando todas foram processadas

        # 9. Controle de Amostras UNKNOWN para reclassificação
        self.unknown_samples = []  # Lista de (idx, sample_data, ground_truth) para reclassificar
        self.final_decisions = {}   # Dicionário idx -> (sample_data, ground_truth, decisao_final)
        self.samples_in_initial_pass = set()  # Índices das amostras processadas na 1ª passagem

        # 10. Rastreamento de clusters para amostras benign
        self.benign_cluster_counts = {}  # cluster_id -> {'count': int, 'final': {label: count}}
        self.benign_cluster_examples = []  # exemplos de inspeção inicial

        print(f"--- {self.node_id.upper()} (GOSSIP NODE) INICIADO ---")
        print(f"Endereço de Escuta: {self.bind_host}:{self.port}")

    def start(self):
        """Inicia o servidor UDP/TCP e a thread de disseminação de gossip."""
        self.server.start_listening()
        self.gossip_thread.start()
        print(f"[{self.node_id.upper()}] Disseminação de Gossip ativada (Intervalo: {self.gossip_interval}s).")

    # ==========================================
    # LÓGICA DE ENVENENAMENTO (POISONING)
    # ==========================================

    def _poisoning_active(self):
        """Ativa o envenenamento após o atraso definido"""
        if self.poison_rate <= 0:
            return False
        return (time.time() - self.start_time) >= self.delay

    def _poison(self, decision: str):
        """Envenena ou não a decisão final (Inverte BENIGN <-> FDI/INTRUSION)"""
        if random.random() >= self.poison_rate:
            return decision  # Escapa do envenenamento pela probabilidade

        print(f"[{self.node_id.upper()}] [AVISO] *** ATAQUE: CLASSIFICACAO ENVENENADA ***")
        return "FDI" if decision == "benign" else "benign"


    # ==========================================
    # LÓGICA DE GOSSIP LEARNING (REDE ASSÍNCRONA)
    # ==========================================

    def _gossip_loop(self):
        """Loop contínuo que envia conhecimento para vizinhos aleatórios periodicamente."""
        while self.gossip_active:
            time.sleep(self.gossip_interval)
            
            # 1. Busca vizinhos disponíveis
            peers = self.peer_manager.get_other_peers()
            if not peers:
                continue
                
            # 2. Escolhe um vizinho aleatoriamente
            target_peer = random.choice(peers)
            
            # 3. Prepara os dados para enviar (ex: últimas amostras rotuladas)
            knowledge_package = self._prepare_knowledge_package()
            
            if knowledge_package:
                self.client.send_gossip(target_peer, knowledge_package)

    def _prepare_knowledge_package(self):
        """Retorna um pacote de GOSSIP contendo amostras de TREINO para compartilhar.

        O nó envia apenas amostras de treino/experiência local, não amostras de teste.
        """
        # Verificamos se o motor já tem dados de treino
        if hasattr(self.engine, 'y_train') and len(self.engine.y_train) > 0:
            # Pega as últimas amostras de treino válidas, filtrando labels inválidas
            valid_indices = [i for i, label in enumerate(self.engine.y_train) if label not in self.INVALID_LABELS]
            if not valid_indices:
                return None
            # Pega as últimas 5 válidas
            n_samples = min(5, len(valid_indices))
            selected_indices = valid_indices[-n_samples:]
            return {
                "payload_type": "TRAINING_SAMPLES",
                "samples": [self.engine.X_train_raw[i].tolist() for i in selected_indices],
                "labels": [self.engine.y_train[i] for i in selected_indices]
            }
        return None

    def _execute_gossip_logic(self, incoming_package):
        """
        Callback executado pelo Servidor quando recebe um pacote Gossip de outro nó.
        Faz o "merge" do conhecimento externo no modelo local.
        """
        samples = incoming_package.get("samples", [])
        labels = incoming_package.get("labels", [])
        origin = incoming_package.get("node_origin", "UNKNOWN")

        if not samples:
            return

        print(f"[{self.node_id.upper()}] Integrou {len(samples)} amostras de TREINO vindas do nó {origin}.")

        # Adiciona as amostras de treino recebidas ao motor
        for s, l in zip(samples, labels):
            if l not in self.INVALID_LABELS:
                sample_raw = np.asarray(s, dtype=float).reshape(-1)
                # Retrain=False para não reconstruir o modelo a cada única amostra
                self.engine.add_training_sample_raw(sample_raw, l, retrain=False)

        # Ao final do lote recebido, reconstrói/treina o pipeline uma única vez
        self.engine.rebuild()


    # ==========================================
    # CONTROLE DE AMOSTRAS PROCESSADAS
    # ==========================================

    def generate_next_sample(self):
        """
        Gera a próxima amostra não processada do conjunto de teste.
        Retorna uma tupla (idx, sample_data, ground_truth).
        Retorna (None, None, None) quando todas as amostras forem processadas.
        """
        # Tenta usar o CSV B (Final Test) primeiro
        if hasattr(self.engine, 'X_final_test_raw') and self.engine.X_final_test_raw is not None and len(self.engine.X_final_test_raw) > 0:
            total_samples = len(self.engine.X_final_test_raw)
            available_indices = [i for i in range(total_samples) if i not in self.processed_samples]

            if not available_indices:
                # Todas as amostras foram processadas
                self.all_samples_processed = True
                print(f"[{self.node_id.upper()}] [OK] TODAS AS {total_samples} AMOSTRAS DO DATASET FORAM PROCESSADAS!")
                return None, None, None

            # Seleciona a próxima amostra (sequencial ou aleatória conforme config)
            if getattr(self, 'randomize_test_order', False):
                idx = self._rand.choice(available_indices)
            else:
                idx = available_indices[0]
            self.processed_samples.add(idx)
            return idx, self.engine.X_final_test_raw[idx], self.engine.y_final_test[idx]

        # Fallback: Se não tiver CSV B configurado, usa o EVAL (CSV A)
        elif hasattr(self.engine, 'X_eval_raw') and self.engine.X_eval_raw is not None and len(self.engine.X_eval_raw) > 0:
            total_samples = len(self.engine.X_eval_raw)
            available_indices = [i for i in range(total_samples) if i not in self.processed_samples]

            if not available_indices:
                # Todas as amostras foram processadas
                self.all_samples_processed = True
                print(f"[{self.node_id.upper()}] [OK] TODAS AS {total_samples} AMOSTRAS DO DATASET FORAM PROCESSADAS!")
                return None, None, None

            # Seleciona a próxima amostra (ordem sequencial)
            idx = available_indices[0]
            self.processed_samples.add(idx)
            return idx, self.engine.X_eval_raw[idx], self.engine.y_eval[idx]

        # Fallback extremo caso o motor não tenha carregado nada
        print(f"[{self.node_id.upper()}] [AVISO] AVISO: Nenhum dataset de teste encontrado!")
        return None, None, None


    # ==========================================
    # LÓGICA DE DETECÇÃO DE TRÁFEGO (INFERÊNCIA)
    # ==========================================

    def check_traffic_and_act(self, sample_idx, sample_data_array, ground_truth, is_reclassification=False, skip_logging=False, skip_learning=False):
        """
        No Gossip, o fluxo de inferência é puramente local e muito mais rápido.
        A "sabedoria da rede" já está embutida nos pesos do motor via _execute_gossip_logic.
        
        Args:
            sample_idx: Índice único da amostra no dataset de teste
            sample_data_array: Dados da amostra
            ground_truth: Rótulo verdadeiro
            is_reclassification: Se é uma reclassificação de UNKNOWN
            skip_logging: Se True, não loga no CSV (usado durante reclassificações iterativas)
        """
        print(f"[{self.node_id.upper()}] Analisando amostra (Ground Truth: {ground_truth})")

        # 1. Faz a inferência usando o motor local (que está constantemente aprendendo com a rede)
        results = self.engine.classify_and_check_conflict(sample_data_array)
        final_decision = results.get("classification", "UNKNOWN")

        # Se a inferência detectar conflito / outlier ou resultar em UNKNOWN, mantenha UNKNOWN.
        if final_decision == "CONFLICT_DETECTED" or final_decision in self.INVALID_LABELS:
            final_decision = "UNKNOWN"

        # 2. Verifica se o nó atual está atuando como malicioso
        if self._poisoning_active():
            final_decision = self._poison(final_decision)

        # 3. Aprende com a própria amostra recém-classificada (Online Learning Pessoal)
        # Em algumas execuções (p. ex. avaliação determinística) queremos desabilitar
        # o aprendizado online para evitar que a ordem das amostras afete os resultados.
        # Somente aprenda online a partir de amostras que NÃO sejam do conjunto de teste.
        # Aprendizado a partir de amostras recebidas via GOSSIP é tratado em _execute_gossip_logic().
        if self.online_learning_enabled and not skip_learning and final_decision not in self.INVALID_LABELS:
            # Permite aprendizado online a partir de amostras classificadas (inclui amostras de teste)
            self.engine.add_training_sample_raw(sample_data_array, final_decision, retrain=True)
        elif not self.online_learning_enabled and not skip_learning:
            print(f"[{self.node_id.upper()}] Online learning está desabilitado; amostra não será adicionada ao treino.")

        # 4. Se for classificação inicial e resultou em UNKNOWN, armazena para reclassificação
        if not is_reclassification and final_decision == "UNKNOWN" and sample_idx is not None:
            self.unknown_samples.append((sample_idx, sample_data_array, ground_truth))

        # 5. Rastreia a decisão final (será logada apenas uma vez ao final)
        if sample_idx is not None:
            self.final_decisions[sample_idx] = (sample_data_array, ground_truth, final_decision)

        # 7. Loga a decisão no CSV APENAS se não estiver em reclassificação ou se não pular logging
        if ground_truth == "benign" and sample_idx is not None:
            cluster_id = results.get("cluster_id", -1)
            entry = self.benign_cluster_counts.setdefault(cluster_id, {"count": 0, "final": {}})
            entry["count"] += 1
            entry["final"][final_decision] = entry["final"].get(final_decision, 0) + 1
            if len(self.benign_cluster_examples) < 20:
                self.benign_cluster_examples.append((sample_idx, cluster_id, final_decision, results.get("decisions", [])))

        if not skip_logging:
            self.logger.log_decisao(ground_truth, final_decision, self.node_id)

        print(f"[{self.node_id.upper()}] Decisão Final (Gossip-driven): {final_decision}")
        
        return final_decision


    def try_reclassify_unknowns(self):
        """
        Tenta reclassificar as amostras que foram marcadas como UNKNOWN.
        Retorna o número de amostras que foram reclassificadas com sucesso.
        
        IMPORTANTE: Não loga no CSV durante reclassificações. O logging final acontece
        apenas uma vez ao término de todas as iterações.
        """
        if not self.unknown_samples:
            return 0

        reclassified_count = 0
        remaining_unknowns = []

        for idx, sample_data, ground_truth in self.unknown_samples:
            # Tenta classificar novamente SEM logar no CSV (skip_logging=True)
            result = self.check_traffic_and_act(
                idx,
                sample_data, 
                ground_truth, 
                is_reclassification=True,
                skip_logging=True,  # Não loga durante reclassificações
                skip_learning=True  # Não aprender durante reclassificações de teste
            )
            
            if result != "UNKNOWN":
                reclassified_count += 1
                print(f"[{self.node_id.upper()}] [REC] Reclassificada amostra {idx}: UNKNOWN -> {result}")
            else:
                remaining_unknowns.append((idx, sample_data, ground_truth))

        self.unknown_samples = remaining_unknowns
        return reclassified_count


    def log_final_decisions(self):
        """
        Loga as decisões finais de TODAS as amostras uma única vez.
        Chamado após todas as iterações de reclassificação serem concluídas.
        """
        total_logged = 0
        
        # Loga todas as decisões armazenadas em final_decisions
        for idx in sorted(self.final_decisions.keys()):
            sample_data, ground_truth, final_decision = self.final_decisions[idx]
            self.logger.log_decisao(ground_truth, final_decision, self.node_id)
            total_logged += 1
        
        print(f"[{self.node_id.upper()}] [OK] Logging final concluido: {total_logged} amostras logadas")
        return total_logged

    def report_benign_cluster_distribution(self):
        total = sum(v["count"] for v in self.benign_cluster_counts.values())
        print(f"[{self.node_id.upper()}] [BENIGN CLUSTER REPORT] {total} benign samples traced.")
        if total == 0:
            print(f"[{self.node_id.upper()}] Nenhum benign processado para rastreamento de cluster.")
            return
        for cluster_id in sorted(self.benign_cluster_counts.keys()):
            stats = self.benign_cluster_counts[cluster_id]
            print(f"  Cluster {cluster_id}: {stats['count']} amostras -> {stats['final']}")
        if self.benign_cluster_examples:
            print(f"[{self.node_id.upper()}] Exemplos de benign analisados (até 20):")
            for sample_idx, cluster_id, final_decision, decisions in self.benign_cluster_examples:
                print(f"    idx={sample_idx} cluster={cluster_id} final={final_decision} votos={decisions}")
