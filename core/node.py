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
        self.gossip_interval = 2 # Frequência de fofoca (em segundos)
        self.gossip_thread = threading.Thread(target=self._gossip_loop, daemon=True)

        # 8. Controle de Amostras Processadas (para parada automática)
        self.processed_samples = set()  # Índices das amostras já processadas
        self.all_samples_processed = False  # Flag para indicar quando todas foram processadas

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

        print(f"[{self.node_id.upper()}] ⚠ *** ATAQUE: CLASSIFICAÇÃO ENVENENADA ***")
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
        """Retorna um dicionário com as últimas amostras aprendidas para compartilhar."""
        # Verificamos se o motor já tem dados de treino
        if hasattr(self.engine, 'y_train') and len(self.engine.y_train) > 0:
            # Pega as últimas 5 amostras que o nó processou
            n_samples = min(5, len(self.engine.y_train))
            return {
                "samples": self.engine.X_train_raw[-n_samples:].tolist(),
                "labels": self.engine.y_train[-n_samples:].tolist()
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

        print(f"[{self.node_id.upper()}] Integrou {len(samples)} amostras vindas do nó {origin}.")

        # Adiciona as amostras ao motor
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
        Retorna None quando todas as amostras foram processadas.
        """
        # Tenta usar o CSV B (Final Test) primeiro
        if hasattr(self.engine, 'X_final_test_raw') and self.engine.X_final_test_raw is not None and len(self.engine.X_final_test_raw) > 0:
            total_samples = len(self.engine.X_final_test_raw)
            available_indices = [i for i in range(total_samples) if i not in self.processed_samples]

            if not available_indices:
                # Todas as amostras foram processadas
                self.all_samples_processed = True
                print(f"[{self.node_id.upper()}] ✅ TODAS AS {total_samples} AMOSTRAS DO DATASET FORAM PROCESSADAS!")
                return None, None

            # Seleciona a próxima amostra (ordem sequencial)
            idx = available_indices[0]
            self.processed_samples.add(idx)
            return self.engine.X_final_test_raw[idx], self.engine.y_final_test[idx]

        # Fallback: Se não tiver CSV B configurado, usa o EVAL (CSV A)
        elif hasattr(self.engine, 'X_eval_raw') and self.engine.X_eval_raw is not None and len(self.engine.X_eval_raw) > 0:
            total_samples = len(self.engine.X_eval_raw)
            available_indices = [i for i in range(total_samples) if i not in self.processed_samples]

            if not available_indices:
                # Todas as amostras foram processadas
                self.all_samples_processed = True
                print(f"[{self.node_id.upper()}] ✅ TODAS AS {total_samples} AMOSTRAS DO DATASET FORAM PROCESSADAS!")
                return None, None

            # Seleciona a próxima amostra (ordem sequencial)
            idx = available_indices[0]
            self.processed_samples.add(idx)
            return self.engine.X_eval_raw[idx], self.engine.y_eval[idx]

        # Fallback extremo caso o motor não tenha carregado nada
        print(f"[{self.node_id.upper()}] ⚠ AVISO: Nenhum dataset de teste encontrado!")
        return None, None


    # ==========================================
    # LÓGICA DE DETECÇÃO DE TRÁFEGO (INFERÊNCIA)
    # ==========================================

    def check_traffic_and_act(self, sample_data_array, ground_truth):
        """
        No Gossip, o fluxo de inferência é puramente local e muito mais rápido.
        A "sabedoria da rede" já está embutida nos pesos do motor via _execute_gossip_logic.
        """
        print(f"[{self.node_id.upper()}] Analisando amostra (Ground Truth: {ground_truth})")

        # 1. Faz a inferência usando o motor local (que está constantemente aprendendo com a rede)
        results = self.engine.classify_and_check_conflict(sample_data_array)
        
        # Como não existe mais a espera por "conselho", usamos a melhor predição do modelo local
        # O classifier_engine no Gossip geralmente deve retornar a decisão majoritária do ensemble.
        best_model_class = self.engine.counseling_logic(sample_data_array)
        final_decision = best_model_class

        # 2. Verifica se o nó atual está atuando como malicioso
        if self._poisoning_active():
            final_decision = self._poison(final_decision)

        # 3. Aprende com a própria amostra recém-classificada (Online Learning Pessoal)
        if final_decision not in self.INVALID_LABELS:
            self.engine.add_training_sample_raw(sample_data_array, final_decision, retrain=True)

        print(f"[{self.node_id.upper()}] Decisão Final (Gossip-driven): {final_decision}")
        
        return final_decision
