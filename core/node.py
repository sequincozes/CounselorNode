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

    def learn_with_advice(self, sample, label):
        # Transforma o sample em linha de uma dimensão (1D), ao invés de matriz 2D
        sample = sample.reshape(1, -1)

        # Adiciona a nova amostra às features de treino
        self.engine.X_train = np.vstack([self.engine.X_train, sample])
        self.engine.y_train = np.append(self.engine.y_train, label)

        #Re-executa os processos do motor de aprendizado
        self.engine.rebuild()
        print("O nó {self.node_id} aprendeu com a amostra {sample} de rótulo {label}.")

    def __init__(self, detected_ip, local_port=None):
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
        self.logger = CounselorLogger(self.node_id, use_log_folder=False)
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

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"Endereço de Escuta: {self.bind_host}:{self.port}")

    def start(self):
        """Inicia o servidor e mantém o nó ativo."""
        self.server.start_listening()

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

                if decision == 'INTRUSION':
                    print(f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho confirmou INTRUSAO.")
                    return "INTRUSION"

                print(f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho decidiu por NORMAL.")
                return "NORMAL"

            return "LOOP_CLOSED"  # Fallback se não houver resposta

        # 8. Se NÃO HÁ conflito local (Decisão de alta confiança do conselheiro)
        else:
            classification = results['classification']
            print(f"[{self.node_id.upper()}] (Conselheiro) Análise local sem conflito. Decisão: {classification}.")

            # Mapeamento simples: se o motor retornou 'NORMAL', enviamos 'NORMAL', caso contrário 'INTRUSION'
            # Nota: Você pode retornar a classe específica (ex: 'DoS') se o seu servidor tratar isso.
            return "NORMAL" if classification == 'NORMAL' else "INTRUSION"
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

            self.learn_with_advice(sample_data_array, final_decision)
            return final_decision

        else:
            # Se não há conflito, usamos a decisão local
            final_decision = "INTRUSION" if classification != 'NORMAL' else "NORMAL"
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {final_decision}.")
            return final_decision