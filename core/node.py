import sys
import time
import numpy as np
import json
import threading
import random  # Necessário para o client.request_counsel se usar random.choice

# Importando camadas
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient
from infrastructure.logger import CounselorLogger
from core.classifier_engine import ClassifierEngine


class CounselorNode:
    """A classe principal que representa o IDS (Detector) na Counselors Network."""

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


    def __init__(self, detected_ip, local_port=None, poison_rate=0.0, delay=0):
        # 1. Configuração (Correto: Passando IP e Porta opcional)
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

        # 4. Motor ML
        ml_config = self.peer_manager.get_ml_config()
        self.engine = ClassifierEngine(ml_config)

        # 5. Cliente de Rede
        self.client = CounselorClient(self.node_id, self.peer_manager, self.logger)

        # 6. Servidor de Rede (Usando a porta correta vinda do config)
        self.server = CounselorServer(
            self.bind_host,
            self.port,
            self.node_id,
            self._execute_counseling_logic,
            self.logger,
            self.peer_manager
        )

        # 7. Poisoning
        self.poison_rate = poison_rate
        self.delay = delay
        self.start_time = time.time()

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"Endereço de Escuta: {self.bind_host}:{self.port}")

    def start(self):
        """Inicia o servidor e mantém o nó ativo."""
        self.server.start_listening()

    def poisoning_active(self):
        """Ativa o envenenamento após o atraso definido"""
        
        if self.poison_rate <= 0:
            return False
        return (time.time() - self.start_time) >= self.delay
    
    def poison(self, decision: str):
        """
        Envenena ou não a decisão final ('NORMAL' ou 'INTRUSION')
        """

        if random.random() >= self.poison_rate:
            return decision  # Não envenenado

        print(f"[{self.node_id.upper()}] ⚠ *** CONSELHO ENVENENADO EMITIDO ***")

        return "INTRUSION" if decision == "NORMAL" else "NORMAL" # Inverte a decisão


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

        # Se o requester_ip e porta (de quem chamou) não estiverem na cadeia, poderiam ser adicionados,
        # mas o padrão é adicionar a si mesmo antes de passar adiante.
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
            all_peers = self.peer_manager.get_other_peers()

            peers_to_ask = [
                p for p in all_peers
                if f"{p['ip']}:{p['port']}" not in current_chain
            ]

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

           # if self.poisoning_active():
            #    classification = self.poison(classification) "AVERIGUANDO LÓGICA"

            # Mapeamento simples: se o motor retornou 'NORMAL', enviamos 'NORMAL', caso contrário 'INTRUSION'
            # Nota: Você pode retornar a classe específica (ex: 'DoS') se o seu servidor tratar isso.
            return classification#"NORMAL" if classification == 'benign' else "INTRUSION"

    def check_traffic_and_act(self, sample_data_array, ground_truth):
        """
        Processa uma amostra de tráfego local e verifica por conflito.
        Inicia a cadeia de requisição se um conflito for detectado.
        """
        # print(f"\n[{self.node_id.upper()}] Analisando amostra (primeiras 25 features): {sample_data_array[:25]}...")
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

            # --- NOVO: Inicializa a cadeia de IPs consultados ---
            local_ip = self.peer_manager.get_local_info().get('ip')
            initial_chain = [local_ip]

            # Busca outros pares para consultar (exclui a si mesmo)
            other_peers = self.peer_manager.get_other_peers()

            if not other_peers:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito detectado, mas não há outros pares. Usando decisão local padrão.")
                # Retorna a decisão local de desempate
                best_model_class = self.engine.counseling_logic(sample_data_array)
                final_decision = best_model_class  # "INTRUSION" if best_model_class != 'benign' else "NORMAL"
                return final_decision

            # O cliente seleciona um par aleatório da lista 'other_peers'
            counsel_response = self.client.request_counsel(
                sample_data_array,
                other_peers,
                ground_truth,
                initial_chain  # Passa a cadeia inicial
            )

            if counsel_response:
                counsel_decision = counsel_response.get('decision')
            else:
                counsel_decision = None  # Falha de conexão inicial

            if counsel_decision == 'LOOP_CLOSED':
                # --- NOVO: Lógica de Resolução de Loop Fechado ---
                print(
                    f"[{self.node_id.upper()}] Ação: LOOP_CLOSED detectado. Usando o classificador local de maior confiança...")

                # 3. Usa a lógica de alta confiança (melhor modelo) do ClassifierEngine para desempate
                best_model_class = self.engine.counseling_logic(sample_data_array)
                final_decision = best_model_class  # "INTRUSION" if best_model_class != 'benign' else "NORMAL"

                print(
                    f"[{self.node_id.upper()}] WARNING: Decisão final: {final_decision} (Melhor Classificador Local). Classe: {best_model_class}")
                # ------------------------------------------------

            # elif counsel_decision == 'INTRUSION':
            #     final_decision = "INTRUSION"
            #     print(
            #         f"[{self.node_id.upper()}] Ação: Decisão Final: {final_decision} (Confirmado por {counsel_response['counselor_id']}).")

            else:
                final_decision = counsel_decision  # "NORMAL"  # Resposta padrão se o conselho não for intrusão/loop
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão Final: {final_decision}"
                    f" (Confirmado por {counsel_response['counselor_id']})."
                )

            counselor_id = "UNKNOWN"
            if counsel_response and isinstance(counsel_response, dict):
                counselor_id = counsel_response.get("counselor_id", "UNKNOWN")

            print(f"Aprendendo com nova amostra de {final_decision}: {sample_data_array}")
            self.learn_with_advice(sample_data_array, final_decision, counselor_id=counselor_id)

            return final_decision

        else:
            # Se não há conflito, usamos a decisão local
            final_decision = classification  # "INTRUSION" if classification != 'benign' else "NORMAL"
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {final_decision}.")
            return final_decision
