import sys
import time
import numpy as np
import json
import threading
import random

# Importando camadas
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient
from infrastructure.sample_networking import SampleReceiver
from infrastructure.logger import CounselorLogger
from core.classifier_engine import ClassifierEngine
from core.reputation_manager import ReputationManager


class CounselorNode:
    """A classe principal que representa o IDS (Detector) na Counselors Network."""

    # --- FILTRO DE LABELS INVÁLIDAS ---
    INVALID_LABELS = {
        "ERROR_CLASSIFICATION",
        "ERROR",
        "UNKNOWN",
        None
    }

    def learn_with_advice(self, sample_raw, label, counselor_id="UNKNOWN"):
        """
        Aprendizado online:
        - adiciona a amostra RAW no TRAIN_RAW
        - re-treina pipeline (rebuild)
        - escreve log de F1 por cluster após a atualização
        """
        # Garante shape 1D (n_features,)
        sample_raw = np.asarray(sample_raw, dtype=float).reshape(-1)

        # 1) adiciona no RAW (não no escalado!)
        if hasattr(self.engine, "add_training_sample_raw"):
            self.engine.add_training_sample_raw(sample_raw, label, retrain=False)
        else:
            # fallback (menos recomendado): atualiza X_train_raw manualmente
            x = sample_raw.reshape(1, -1)
            self.engine.X_train_raw = np.vstack([self.engine.X_train_raw, x])
            self.engine.y_train = np.append(self.engine.y_train, label)

        # 2) refaz todo o pipeline
        self.engine.rebuild()

        # 3) snapshot de F1 por cluster e log
        if hasattr(self.engine, "get_cluster_f1_snapshot_rows"):
            rows = self.engine.get_cluster_f1_snapshot_rows()
            if hasattr(self.logger, "log_cluster_f1_snapshot"):
                self.logger.log_cluster_f1_snapshot(
                    node_id=self.node_id,
                    rows=rows,
                    event="ADVICE_LEARN",
                    sample_label=str(label),
                    counselor_id=str(counselor_id)
                )

        print(f"[{self.node_id.upper()}] Aprendeu com conselho. label={label} counselor={counselor_id}")



    def __init__(self, detected_ip, local_port=None, poison_rate=1, delay=0,
                 defense_tau=0.6, defense_min_challenges=10):
        # Componente 1: Configuração (Inicializado com o IP detectado)
        self.peer_manager = ConfigManager(detected_ip, local_port=local_port)
        local_info = self.peer_manager.get_local_info()

        # 2. Informações lidas do config (Centralizado no peer_manager)
        self.node_id = self.peer_manager.node_id
        self.port = self.peer_manager.local_port
        self.local_ip = detected_ip
        self.config_ip = local_info.get('ip')
        self.bind_host = '0.0.0.0'

        # 3. Logger
        self.logger = CounselorLogger(self.node_id, use_log_folder=True)
        print(f"Logger inicializado para o nó: {self.node_id}")

        # Taxa de envenenamento
        self.poison_rate = poison_rate

        # Tempo de início do envenenamento
        self.delay = delay
        self.start_time = time.time()

        # 4. Motor ML
        ml_config = self.peer_manager.get_ml_config()
        self.engine = ClassifierEngine(ml_config)

        # 5. Mecanismo de Defesa: Reputação
        all_peers = self.peer_manager.get_other_peers()
        self.reputation_manager = ReputationManager(
            peers=all_peers,
            tau=defense_tau,
            min_challenges=defense_min_challenges,
            logger=self.logger,
            node_id=self.node_id,
        )

        # 6. Cliente de Rede (com referência ao ReputationManager)
        self.client = CounselorClient(
            self.node_id,
            self.peer_manager,
            self.logger,
            reputation_manager=self.reputation_manager,
        )

        # 7. Servidor de Rede (Usando a porta correta vinda do config)
        self.server = CounselorServer(
            self.bind_host,
            self.port,
            self.node_id,
            self._execute_counseling_logic,
            self.logger,
            self.peer_manager
        )

        # 8. Receptor de Amostras (novo canal, porta separada)
        self.sample_port = self.peer_manager.get_local_info().get('sample_port', self.port + 1000)
        self.sample_receiver = SampleReceiver(
            self.bind_host,
            self.sample_port,
            self.node_id,
            self.check_traffic_and_act,   # reaproveita 100% o método já existente
            self.logger
        )

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"Endereço de Escuta: {self.bind_host}:{self.port}")
        print(f"Endereço de Escuta (amostras): {self.bind_host}:{self.sample_port}")

    def start(self):
        """Inicia os servidores (conselhos e amostras) e mantém o nó ativo."""
        self.server.start_listening()
        self.sample_receiver.start_listening()

    def _poisoning_active(self):
        """Ativa o envenenamento após o atraso definido"""
        print("POISONING ATIVADO!")
        if self.poison_rate <= 0:
            return False
        return (time.time() - self.start_time) >= self.delay

    def _poison(self, decision: str):
        """
        Envenena ou não a decisão final ('NORMAL' ou 'INTRUSION')
        """
        if random.random() >= self.poison_rate:
            return decision  # Não envenenado

        print(f"[{self.node_id.upper()}] ⚠ *** CONSELHO ENVENENADO EMITIDO ***")

        return "FDI" if decision == "benign" else "benign"  # Inverte a decisão


    def _execute_counseling_logic(self, sample_data_array, requester_id=None, requester_ip=None,
                                  ground_truth="N/A_from_peer", requester_chain=None):
        """
        Função de callback executada quando este nó recebe um pedido de aconselhamento.
        Implementa a lógica de conselho em cascata usando IP:PORTA para evitar loops.
        """
        # 1. Identificação única deste nó (IP:Porta) para a cadeia
        local_info = self.peer_manager.get_local_info()
        my_identity = f"{local_info.get('ip')}:{local_info.get('port')}"

        print(f"\n[{self.node_id.upper()}] (Conselheiro) Recebeu pedido de {requester_id} ({requester_ip}).")

        # 2. Inicializa e atualiza a cadeia de requisição
        if requester_chain is None:
            requester_chain = []

        current_chain = requester_chain + [my_identity]

        print(f"[{self.node_id.upper()}] (Conselheiro) Cadeia de requisição atual: {current_chain}")

        # 3. Executa a lógica de conflito local
        results = self.engine.classify_and_check_conflict(sample_data_array)
        conflict = results['conflict']

        # 4. Caso haja conflito interno no conselheiro, ele busca um terceiro nó
        if conflict:
            print(
                f"[{self.node_id.upper()}] (Conselheiro) CONFLITO INTERNO ao aconselhar {requester_id}. Buscando ajuda externa...")

            # 5. Filtra pares: EXCLUI nós que já estão na cadeia (comparando IP:Porta)
            #    E também pares com reputação abaixo do limiar (ISOLATED)
            all_peers = self.peer_manager.get_other_peers()

            peers_filtered_chain = [
                p for p in all_peers
                if f"{p['ip']}:{p['port']}" not in current_chain
            ]

            # Aplica filtro de reputação (remove pares ISOLATED)
            peers_to_ask = self.reputation_manager.get_active_peers(peers_filtered_chain)

            if len(peers_to_ask) < len(peers_filtered_chain):
                n_removed = len(peers_filtered_chain) - len(peers_to_ask)
                print(f"[{self.node_id.upper()}] (Defesa) {n_removed} par(es) ignorado(s) por estarem ISOLATED.")

            # LÓGICA DE LOOP CLOSED: Se não houver ninguém novo para perguntar
            if not peers_to_ask:
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) ALERTA: Loop Fechado! Todos os pares IP:Porta já consultados. Retornando LOOP_CLOSED.")
                return "LOOP_CLOSED"

            # 6. Pede conselho ao próximo par disponível
            counsel_response = self.client.request_counsel(
                sample_data_array,
                peers_to_ask,
                ground_truth,
                current_chain  # Passa a cadeia atualizada com este nó incluso
            )

            # 7. Trata a resposta do terceiro nó
            if counsel_response:
                decision = counsel_response.get('decision')

                # Se o próximo nó falhou ou também detectou loop, propagamos o LOOP_CLOSED
                if decision in ['LOOP_CLOSED', 'ERROR_CONNECTION', 'ERROR_TIMEOUT', 'ERROR_CONNECTION_REFUSED']:
                    print(
                        f"[{self.node_id.upper()}] (Conselheiro) Próximo peer reportou falha/loop. Propagando LOOP_CLOSED.")
                    return "LOOP_CLOSED"

                print(f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho confirmou {decision}.")
                return decision

            return "LOOP_CLOSED"  # Fallback se não houver resposta

        # 8. Se NÃO HÁ conflito local (Decisão de alta confiança do conselheiro)
        else:
            classification = results['classification']
            print(f"[{self.node_id.upper()}] (Conselheiro) Análise local sem conflito. Decisão: {classification}.")

            if self._poisoning_active():
                classification = self._poison(classification)  # envenenamento de decisão

            return classification


    def check_traffic_and_act(self, sample_data_array, ground_truth):
        """
        Processa uma amostra de tráfego local e verifica por conflito.
        Inicia a cadeia de requisição se um conflito for detectado.
        Atualiza a reputação dos conselheiros quando o ground truth é conhecido.
        """
        print(f"[{self.node_id.upper()}] (Ground Truth para esta amostra: {ground_truth})")

        # 1. Classifica e verifica por conflito usando o motor DCS
        results = self.engine.classify_and_check_conflict(sample_data_array)

        classification = results['classification']
        conflict = results['conflict']
        decisions = results['decisions']
        cluster_id = results.get('cluster_id', 'N/A')

        print(f"[{self.node_id.upper()}] Resultado DCS Local (Cluster {cluster_id}): {classification}")
        print(f"[{self.node_id.upper()}] Decisões Locais (Classes): {decisions}")

        # 2. Lógica de Gatilho da Rede de Conselheiros
        if conflict:
            print(
                f"[{self.node_id.upper()}] Alerta: CONFLITO DE CLASSIFICADOR DETECTADO! Consultando a Counselors Network.")

            local_info = self.peer_manager.get_local_info()
            initial_chain = [f"{local_info.get('ip')}:{local_info.get('port')}"]

            # Busca outros pares, filtrando pares ISOLATED
            other_peers_all = self.peer_manager.get_other_peers()
            other_peers = self.reputation_manager.get_active_peers(other_peers_all)

            if len(other_peers) < len(other_peers_all):
                n_blocked = len(other_peers_all) - len(other_peers)
                print(f"[{self.node_id.upper()}] (Defesa) {n_blocked} par(es) ISOLATED ignorado(s) na consulta.")

            if not other_peers:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito detectado, mas não há outros pares disponíveis. Usando decisão local padrão.")
                best_model_class = self.engine.counseling_logic(sample_data_array)
                final_decision = best_model_class
                return final_decision

            # O cliente seleciona um par aleatório da lista filtrada
            counsel_response = self.client.request_counsel(
                sample_data_array,
                other_peers,
                ground_truth,
                initial_chain
            )

            if counsel_response:
                counsel_decision = counsel_response.get('decision')
            else:
                counsel_decision = None  # Falha de conexão inicial

            if counsel_decision == 'LOOP_CLOSED':
                print(
                    f"[{self.node_id.upper()}] Ação: LOOP_CLOSED detectado. Usando o classificador local de maior confiança...")

                best_model_class = self.engine.counseling_logic(sample_data_array)
                final_decision = best_model_class

                print(
                    f"[{self.node_id.upper()}] WARNING: Decisão final: {final_decision} (Melhor Classificador Local). Classe: {best_model_class}")

            else:
                final_decision = counsel_decision
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão Final: {final_decision}"
                    f" (Confirmado por {counsel_response['counselor_id'] if counsel_response else 'N/A'})."
                )

            # --- Atualização de reputação quando ground_truth é conhecido ---
            if counsel_response and isinstance(counsel_response, dict):
                counselor_id = counsel_response.get("counselor_id", "UNKNOWN")
                counselor_ip = counsel_response.get("counselor_ip", None)
                counselor_port = counsel_response.get("counselor_port", None)

                gt_str = str(ground_truth)
                is_valid_gt = gt_str not in ("N/A", "N/A_from_peer", "UNKNOWN", "")

                if is_valid_gt and counselor_ip and counselor_port:
                    dec_str = str(counsel_decision) if counsel_decision else ""
                    self.reputation_manager.register_challenge_result(
                        peer_ip=counselor_ip,
                        peer_port=int(counselor_port),
                        peer_decision=dec_str,
                        ground_truth=gt_str,
                        latencia_ms=counsel_response.get("latencia_ms", 0.0),
                    )
                    print(
                        f"[{self.node_id.upper()}] (Defesa) Reputação de {counselor_id} atualizada. "
                        f"decisão={dec_str} | gt={gt_str}"
                    )
            else:
                counselor_id = "UNKNOWN"

            if final_decision not in self.INVALID_LABELS:
                print(f"Aprendendo com nova amostra de {final_decision}: {sample_data_array}")
                self.learn_with_advice(sample_data_array, final_decision, counselor_id=counselor_id)

            return final_decision

        else:
            # Se não há conflito, usamos a decisão local
            final_decision = classification
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {final_decision}.")
            return final_decision
