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

    def learnWithConflict(self, sample, label):
        sample.
        self.X_train = np.vstack([self.X_train, sample])
        self.engine._apply_clustering()
        self.engine._train_dcs_model()
        pass

    def __init__(self, detected_ip):
        # Componente 1: Configuração (Inicializado com o IP detectado)
        self.peer_manager = ConfigManager(detected_ip)
        local_info = self.peer_manager.get_local_info()

        # Informações lidas do config
        self.port = local_info.get('port')
        self.node_id = local_info.get('name')
        self.config_ip = local_info.get('ip')  # O IP que outros pares usam para ME encontrar

        # O host de "bind" é 0.0.0.0 para escutar em todas as interfaces
        self.bind_host = '0.0.0.0'

        # Componente 1.5: Logger
        self.logger = CounselorLogger(self.node_id, use_log_folder=False)
        print(f"Logger inicializado. Logs serão salvos na raiz do projeto.")

        # Componente 2: Motor ML (Inteligência IDS)
        ml_config = self.peer_manager.get_ml_config()
        self.engine = ClassifierEngine(ml_config)

        # Componente 3 & 4: Rede
        self.client = CounselorClient(self.node_id, self.peer_manager, self.logger)

        # O servidor recebe o callback para a lógica de aconselhamento
        self.server = CounselorServer(
            self.bind_host,
            self.port,
            self.node_id,
            self._execute_counseling_logic,  # <- CALLBACK
            self.logger,
            self.peer_manager
        )

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"Endereço: {self.config_ip}:{self.port}")

    def start(self):
        """Inicia o servidor e mantém o nó ativo."""
        self.server.start_listening()


    def _execute_counseling_logic(self, sample_data_array, requester_id=None, requester_ip=None,
                                  ground_truth="N/A_from_peer", requester_chain=None):
        """
        Função de callback executada quando este nó recebe um pedido de aconselhamento.
        Implementa a lógica de conselho em cascata, evitando loops.
        """
        print(f"\n[{self.node_id.upper()}] (Conselheiro) Recebeu pedido de {requester_id}. Analisando amostra...")

        # 1. Atualiza a cadeia de requisição (Adiciona o IP deste nó)
        if requester_chain is None:
            requester_chain = []

        local_ip = self.peer_manager.get_local_info().get('ip')
        current_chain = requester_chain + [local_ip]

        print(f"[{self.node_id.upper()}] (Conselheiro) Cadeia de requisição atual: {current_chain}")

        # 2. Executa a lógica de conflito local
        results = self.engine.classify_and_check_conflict(sample_data_array)
        conflict = results['conflict']

        # 3. Verifica se HÁ conflito
        if conflict:
            print(
                f"[{self.node_id.upper()}] (Conselheiro) CONFLITO INTERNO detectado ao tentar aconselhar {requester_id}. Solicitando um segundo conselho...")

            # 4. Filtra pares: EXCLUI todos os IPs na cadeia de requisição
            all_peers = self.peer_manager.get_other_peers()

            # Remove qualquer par cujo IP já está na cadeia
            peers_to_ask = [
                p for p in all_peers
                if p['ip'] not in current_chain
            ]

            # --- LÓGICA DE LOOP CLOSED (NOVO) ---
            if not peers_to_ask:
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) ALERTA: Loop Fechado! Todos os pares já foram consultados na cadeia. Retornando LOOP_CLOSED.")
                return "LOOP_CLOSED"

            # 5. Pede conselho a um terceiro
            counsel_response = self.client.request_counsel(
                sample_data_array,
                peers_to_ask,
                ground_truth,
                current_chain  # Passa a cadeia atualizada
            )

            # 6. Baseia a resposta final no segundo conselho
            # Propaga o LOOP_CLOSED se o terceiro nó também o retornar
            if counsel_response and counsel_response['decision'] in ['LOOP_CLOSED', 'ERROR_CONNECTION', 'ERROR_TIMEOUT',
                                                                     'ERROR_CONNECTION_REFUSED']:
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) O último peer reportou falha/LOOP_CLOSED. Propagando alerta.")
                return "LOOP_CLOSED"  # Resposta final para o solicitante original (A)

            if counsel_response and counsel_response['decision'] == 'INTRUSION':
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho ({counsel_response['counselor_id']}) confirmou INTRUSAO.")
                return "INTRUSION"  # Resposta final para o solicitante original (A)
            else:
                decision_txt = counsel_response['decision'] if counsel_response else "NENHUM"
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho foi {decision_txt}. Respondendo NORMAL como default.")
                return "NORMAL"  # Resposta final para o solicitante original (A)

        # 3b. Se NÃO HÁ conflito local (Decisão de alta confiança)
        else:
            classification = results['classification']
            print(f"[{self.node_id.upper()}] (Conselheiro) Análise local sem conflito. Decisão: {classification}.")

            # Retorna a classificação local (0, 1, 2) mapeada para "NORMAL" ou "INTRUSION"
            if classification == 'NORMAL':  # Assumindo que '0' é NORMAL
                return "NORMAL"
            else:
                # Retorna a classe específica (ex: '1') que será mapeada para INTRUSION pelo Servidor
                return classification

    def check_traffic_and_act(self, sample_data_array, ground_truth):
        """
        Processa uma amostra de tráfego local e verifica por conflito.
        Inicia a cadeia de requisição se um conflito for detectado.
        """
        print(f"\n[{self.node_id.upper()}] Analisando amostra (primeiras 5 features): {sample_data_array[:5]}...")
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
                final_decision = "INTRUSION" if best_model_class != 'NORMAL' else "NORMAL"
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
                final_decision = "INTRUSION" if best_model_class != 'NORMAL' else "NORMAL"

                print(
                    f"[{self.node_id.upper()}] WARNING: Decisão final: {final_decision} (Melhor Classificador Local). Classe: {best_model_class}")
                # ------------------------------------------------

            elif counsel_decision == 'INTRUSION':
                final_decision = "INTRUSION"
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão Final: {final_decision} (Confirmado por {counsel_response['counselor_id']}).")

            else:
                final_decision = "NORMAL"  # Resposta padrão se o conselho não for intrusão/loop
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito resolvido ou sem conselho externo definitivo. Usando decisão final: {final_decision}.")

            learnWithConflict(sample_data_array, final_decision)
            return final_decision

        else:
            # Se não há conflito, usamos a decisão local
            final_decision = "INTRUSION" if classification != 'NORMAL' else "NORMAL"
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {final_decision}.")
            return final_decision