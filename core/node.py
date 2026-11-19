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
from core.reputation_manager import ReputationManager, is_operational_label


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



    def __init__(self, detected_ip, local_port=None, poison_rate=None, delay=None,
                 defense_enabled=None,
                 defense_tau=None, defense_min_challenges=None,
                 defense_probe_every=None, defense_comeback_threshold=None,
                 challenge_min=None, challenge_max=None,
                 challenge_on_conflict=None):
        # Componente 1: Configuração (Inicializado com o IP detectado)
        self.peer_manager = ConfigManager(detected_ip, local_port=local_port)
        local_info = self.peer_manager.get_local_info()
        ml_config = self.peer_manager.get_ml_config()

        # 2. Informações lidas do config (Centralizado no peer_manager)
        self.node_id = self.peer_manager.node_id
        self.port = self.peer_manager.local_port
        self.local_ip = detected_ip
        self.config_ip = local_info.get('ip')
        self.bind_host = '0.0.0.0'

        # 3. Logger
        self.logger = CounselorLogger(self.node_id, use_log_folder=True)
        print(f"Logger inicializado para o nó: {self.node_id}")

        # Taxa de envenenamento e delay: CLI -> local_info -> ml_config -> default
        if poison_rate is not None:
            self.poison_rate = float(poison_rate)
        else:
            self.poison_rate = float(local_info.get("poison_rate", ml_config.get("poison_rate", 0.0)))

        if delay is not None:
            self.delay = int(delay)
        else:
            self.delay = int(local_info.get("delay", local_info.get("poison_delay", ml_config.get("poison_delay", 0))))
        self.start_time = time.time()

        # 4. Motor ML
        self.engine = ClassifierEngine(ml_config)

        # 5. Mecanismo de Defesa: Reputação
        # Precedência: CLI -> local_info -> ml_config -> default
        if defense_enabled is not None:
            self.defense_enabled = bool(defense_enabled)
        else:
            self.defense_enabled = bool(local_info.get("defense_enabled", ml_config.get("defense_enabled", True)))

        self.defense_tau = float(
            defense_tau if defense_tau is not None
            else local_info.get("defense_tau", ml_config.get("defense_tau", 0.6))
        )
        self.defense_min_challenges = int(
            defense_min_challenges if defense_min_challenges is not None
            else local_info.get("defense_min_challenges", ml_config.get("defense_min_challenges", 10))
        )
        self.defense_probe_every = int(
            defense_probe_every if defense_probe_every is not None
            else local_info.get("defense_probe_every", ml_config.get("defense_probe_every", 5))
        )
        self.defense_comeback_threshold = int(
            defense_comeback_threshold if defense_comeback_threshold is not None
            else local_info.get("defense_comeback_threshold", ml_config.get("defense_comeback_threshold", 3))
        )

        if self.defense_enabled:
            all_peers = self.peer_manager.get_other_peers()
            self.reputation_manager = ReputationManager(
                peers=all_peers,
                tau=self.defense_tau,
                min_challenges=self.defense_min_challenges,
                probe_every=self.defense_probe_every,
                comeback_threshold=self.defense_comeback_threshold,
                logger=self.logger,
                node_id=self.node_id,
            )
        else:
            self.reputation_manager = None
            print(f"[{self.node_id.upper()}] Mecanismo de Defesa: DESATIVADO (sem avaliação de reputação ou isolamento).")

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
        self.sample_port = local_info.get('sample_port', self.port + 1000)
        self.sample_receiver = SampleReceiver(
            self.bind_host,
            self.sample_port,
            self.node_id,
            self.check_traffic_and_act,   # reaproveita 100% o método já existente
            self.logger
        )

        # 9. Desafio Proativo — disparado por contagem de amostras recebidas
        # Precedência: CLI -> local_info -> ml_config -> default
        self._challenge_min_interval = int(
            challenge_min if challenge_min is not None
            else local_info.get("challenge_min_interval", ml_config.get("challenge_min_interval", 5))
        )
        self._challenge_max_interval = int(
            challenge_max if challenge_max is not None
            else local_info.get("challenge_max_interval", ml_config.get("challenge_max_interval", 15))
        )
        self.challenge_on_conflict = bool(
            challenge_on_conflict if challenge_on_conflict is not None
            else local_info.get("challenge_on_conflict", ml_config.get("challenge_on_conflict", False))
        )
        self._samples_since_last_challenge: int = 0
        self._next_challenge_at: int = random.randint(
            self._challenge_min_interval, self._challenge_max_interval
        )
        self._challenge_lock = threading.Lock()
        if self.reputation_manager is not None:
            print(
                f"[{self.node_id.upper()}] Defesa configurada: "
                f"τ={self.defense_tau} | min_challenges={self.defense_min_challenges} | "
                f"probe_every={self.defense_probe_every} | comeback_threshold={self.defense_comeback_threshold}"
            )
            print(
                f"[{self.node_id.upper()}] Desafios de Reputação: "
                f"intervalo=[{self._challenge_min_interval}, {self._challenge_max_interval}] amostras recebidas | "
                f"próximo em {self._next_challenge_at} amostras | "
                f"desafios por conflito: {'ATIVO' if self.challenge_on_conflict else 'DESATIVADO (apenas por amostras)'}"
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

            # Aplica filtro de reputação (remove pares ISOLATED se defesa estiver ativa)
            if self.reputation_manager is not None:
                peers_to_ask = self.reputation_manager.get_active_peers(peers_filtered_chain)
                if len(peers_to_ask) < len(peers_filtered_chain):
                    n_removed = len(peers_filtered_chain) - len(peers_to_ask)
                    print(f"[{self.node_id.upper()}] (Defesa) {n_removed} par(es) ignorado(s) por estarem ISOLATED.")
            else:
                peers_to_ask = peers_filtered_chain

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

    def _send_active_challenge(self, sample_data_array, ground_truth):
        """
        Envia um desafio proativo a um par aleatório para avaliar sua reputação.
        Disparado por contagem de amostras recebidas (independente de conflito).
        Roda em thread daemon separada para não bloquear a resposta ao SampleSender.
        """
        if self.reputation_manager is None:
            return  # Defesa desativada — sem desafios proativos

        all_peers = self.peer_manager.get_other_peers()
        if not all_peers:
            return

        # Sorteia um par aleatório entre TODOS os outros pares conhecidos.
        # Pares isolados também continuam recebendo desafios para avaliar possível recuperação (comeback).
        target = random.choice(all_peers)
        local_info = self.peer_manager.get_local_info()
        chain = [f"{local_info.get('ip')}:{local_info.get('port')}"]

        print(
            f"[{self.node_id.upper()}] (Desafio Ativo) → {target['name']} "
            f"({target['ip']}:{target['port']}) | gt={ground_truth}"
        )

        response = self.client.request_counsel(sample_data_array, [target], ground_truth, chain)

        if response:
            dec = response.get("decision", "")
            c_ip = response.get("counselor_ip", target["ip"])
            c_port = response.get("counselor_port", target["port"])
            gt_str = str(ground_truth)

            if not is_operational_label(dec) and gt_str not in ("N/A", "N/A_from_peer", "UNKNOWN", ""):
                evento = self.reputation_manager.register_challenge_result(
                    peer_ip=c_ip,
                    peer_port=int(c_port),
                    peer_decision=dec,
                    ground_truth=gt_str,
                    latencia_ms=response.get("latencia_ms", 0.0),
                    tipo_evento_override="ACTIVE_CHALLENGE",
                )
                # Consome probe pendente caso o par esteja ISOLATED
                self.reputation_manager.consume_probe_for(c_ip, int(c_port))
                if evento is not None:
                    print(
                        f"[{self.node_id.upper()}] (Desafio Ativo) Reputação de "
                        f"{target['name']} atualizada [ACTIVE_CHALLENGE]: "
                        f"dec={dec} | gt={gt_str} | "
                        f"rep={evento['reputacao']:.4f} | estado={evento['estado']}"
                    )
            else:
                print(
                    f"[{self.node_id.upper()}] (Desafio Ativo) Resposta de {target['name']} "
                    f"ignorada (label operacional ou gt inválido): dec='{dec}'"
                )
        else:
            print(
                f"[{self.node_id.upper()}] (Desafio Ativo) Sem resposta de "
                f"{target['name']} — desafio descartado."
            )

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

            # Busca outros pares, filtrando pares ISOLATED se defesa ativa
            other_peers_all = self.peer_manager.get_other_peers()
            if self.reputation_manager is not None:
                other_peers = self.reputation_manager.get_active_peers(other_peers_all)
                if len(other_peers) < len(other_peers_all):
                    n_blocked = len(other_peers_all) - len(other_peers)
                    print(f"[{self.node_id.upper()}] (Defesa) {n_blocked} par(es) ISOLATED ignorado(s) na consulta.")
            else:
                other_peers = other_peers_all

            if not other_peers:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito detectado, mas não há outros pares disponíveis. Usando decisão local padrão.")
                best_model_class = self.engine.counseling_logic(sample_data_array)
                final_decision = best_model_class
            else:
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

                # --- Atualização de reputação por conflito (opcional, desativada por padrão) ---
                # Quando desativada (padrão), o conselho é usado apenas para desempate operacional
                # e os desafios de reputação ocorrem exclusivamente pelo ciclo de 5 a 15 amostras.
                if self.challenge_on_conflict and self.reputation_manager is not None and counsel_response and isinstance(counsel_response, dict):
                    counselor_id = counsel_response.get("counselor_id", "UNKNOWN")
                    counselor_ip = counsel_response.get("counselor_ip", None)
                    counselor_port = counsel_response.get("counselor_port", None)

                    gt_str = str(ground_truth)
                    is_valid_gt = gt_str not in ("N/A", "N/A_from_peer", "UNKNOWN", "")
                    dec_str = str(counsel_decision) if counsel_decision else ""

                    # Só registra desafio se: ground truth é válido E decisão é uma
                    # label de CLASSE real (não uma label operacional como LOOP_CLOSED).
                    if is_valid_gt and counselor_ip and counselor_port and not is_operational_label(dec_str):
                        evento = self.reputation_manager.register_challenge_result(
                            peer_ip=counselor_ip,
                            peer_port=int(counselor_port),
                            peer_decision=dec_str,
                            ground_truth=gt_str,
                            latencia_ms=counsel_response.get("latencia_ms", 0.0),
                        )
                        if evento is not None:
                            print(
                                f"[{self.node_id.upper()}] (Defesa) Reputação de {counselor_id} atualizada "
                                f"[{evento['tipo_evento']}]. decisão={dec_str} | gt={gt_str} | "
                                f"rep={evento['reputacao']:.4f} | estado={evento['estado']}"
                            )
                            # Marca probe consumido se for um desafio sobre par isolado
                            if counselor_ip and counselor_port:
                                self.reputation_manager.consume_probe_for(counselor_ip, int(counselor_port))
                    elif is_valid_gt and is_operational_label(dec_str):
                        print(
                            f"[{self.node_id.upper()}] (Defesa) Label operacional ignorada na avaliação de "
                            f"{counselor_id}: '{dec_str}' — reputação não alterada."
                        )
                elif counsel_response and isinstance(counsel_response, dict):
                    counselor_id = counsel_response.get("counselor_id", "UNKNOWN")
                else:
                    counselor_id = "UNKNOWN"

                if final_decision not in self.INVALID_LABELS:
                    print(f"Aprendendo com nova amostra de {final_decision}: {sample_data_array}")
                    self.learn_with_advice(sample_data_array, final_decision, counselor_id=counselor_id)

        else:
            # Se não há conflito, usamos a decisão local
            final_decision = classification
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {final_decision}.")

        # --- Desafio Proativo por Amostras Recebidas ---
        # Disparado a cada intervalo aleatório de 5 a 15 amostras recebidas,
        # garantindo que nós sem conflitos também avaliem seus pares regularmente.
        if self.reputation_manager is not None:
            with self._challenge_lock:
                self._samples_since_last_challenge += 1
                if self._samples_since_last_challenge >= self._next_challenge_at:
                    self._samples_since_last_challenge = 0
                    self._next_challenge_at = random.randint(
                        self._challenge_min_interval, self._challenge_max_interval
                    )
                    print(
                        f"[{self.node_id.upper()}] 🎯 Limiar de amostras atingido. "
                        f"Disparando desafio proativo (próximo em {self._next_challenge_at} amostras)..."
                    )
                    threading.Thread(
                        target=self._send_active_challenge,
                        args=(sample_data_array, ground_truth),
                        daemon=True,
                    ).start()

        return final_decision
