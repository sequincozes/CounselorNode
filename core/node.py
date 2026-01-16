import sys
import time
import numpy as np
import json
import random

# Importando camadas
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient
from infrastructure.logger import CounselorLogger  # Importa o novo logger
from core.classifier_engine import ClassifierEngine


class CounselorNode:
    """A classe principal que representa o IDS (Detector) na Counselors Network."""

    def __init__(self, detected_ip, poison_rate=0.0, delay=0):
        # Componente 1: Configuração (Inicializado com o IP detectado)
        self.peer_manager = ConfigManager(detected_ip)
        local_info = self.peer_manager.get_local_info()

        # Informações lidas do config
        self.port = local_info.get('port')
        self.node_id = local_info.get('name')
        self.config_ip = local_info.get('ip')  # O IP que outros pares usam para ME encontrar
        node_dataset = local_info.get('training_dataset')  # Dataset específico deste nó
        # O host de "bind" é 0.0.0.0 para escutar em todas as interfaces
        self.bind_host = '0.0.0.0'

        #Taxa de envenamento
        self.poison_rate = poison_rate

        #Tempo de início do envenamento
        self.delay = delay
        self.start_time = time.time()

        # Componente 1.5: Logger
        # --- CORREÇÃO AQUI ---
        # A chamada agora passa 'use_log_folder=False' para salvar na RAIZ.
        # O logger.py (acima) agora aceita este argumento.
        self.logger = CounselorLogger(self.node_id, use_log_folder=False)
        print(f"Logger inicializado. Logs serão salvos na raiz do projeto.")

        # Componente 2: Motor ML (Inteligência IDS)
        self.engine = ClassifierEngine(self.peer_manager.get_ml_config(), node_dataset )  # Inicializa o treinamento DCS

        # Componente 3 & 4: Rede (Passa o logger e o peer_manager)
        self.client = CounselorClient(self.node_id, self.peer_manager, self.logger)
        self.server = CounselorServer(
            self.bind_host,  # Escuta em 0.0.0.0
            self.port,  # Usa a porta encontrada no config
            self.node_id,
            self._execute_counseling_logic,  # Passa a lógica de decisão como callback
            self.logger,  # Injeta o logger
            self.peer_manager  # Injeta o peer_manager para IP local
        )

        print(f"--- {self.node_id.upper()} INICIADO ---")
        print(f"IP de Configuração (Encontrado via {detected_ip}): {self.config_ip}:{self.port}")
        print(f"Endereço de Bind (Escutando em):             {self.bind_host}:{self.port}")
        print("-" * 30)

    def start(self):
        """Inicia o servidor e mantém o nó ativo."""
        self.server.start_listening()

    def _poisoning_active(self):
        """Ativa o envenenamento após o atraso definido"""
        
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

        return "INTRUSION" if decision == "NORMAL" else "NORMAL" # Inverte a decisão

    
    def _execute_counseling_logic(self, sample_data_array, requester_id=None, requester_ip=None,
                                  ground_truth="N/A_from_peer"):
        """
        Função de callback executada quando este nó recebe um pedido de aconselhamento.
        Se ESTE nó tiver um conflito, ele pedirá um segundo conselho antes de responder.
        """
        print(f"\n[{self.node_id.upper()}] (Conselheiro) Recebeu pedido de {requester_id}. Analisando amostra...")

        # 1. Executa a lógica de conflito local
        results = self.engine.classify_and_check_conflict(sample_data_array)

        classification = results['classification']
        conflict = results['conflict']

        # 2. Verifica se HÁ conflito
        if conflict:
            print(
                f"[{self.node_id.upper()}] (Conselheiro) CONFLITO INTERNO detectado ao tentar aconselhar {requester_id}.")
            print(f"[{self.node_id.upper()}] (Conselheiro) Solicitando um segundo conselho de outro par...")

            # 3. Busca pares, EXCLUINDO o solicitante original
            all_other_peers = self.peer_manager.get_other_peers()

            # Filtra o par que fez a solicitação original (não podemos perguntar a ele)
            peers_to_ask = [
                p for p in all_other_peers
                if p['name'] != requester_id
            ]

            if not peers_to_ask:
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) Não há outros pares (excluindo {requester_id}) para perguntar.")
                # Retorna 'UNKNOWN' para indicar que não podemos resolver.
                return "UNKNOWN"  # Retorno final para o solicitante original

            # 4. Pede conselho a um terceiro
            counsel_response = self.client.request_counsel(
                sample_data_array,
                peers_to_ask,
                ground_truth  # Passa o ground_truth original para o log do próximo nó
            )

            # 5. Baseia a resposta final no segundo conselho
            if counsel_response and counsel_response['decision'] == 'INTRUSION':
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho ({counsel_response['counselor_id']}) confirmou INTRUSAO.")
                return "INTRUSION"  # Resposta final para o solicitante original (A)
            else:
                decision_txt = counsel_response['decision'] if counsel_response else "NENHUM"
                print(
                    f"[{self.node_id.upper()}] (Conselheiro) Segundo conselho foi {decision_txt}. Respondendo NORMAL.")
                return "NORMAL"  # Resposta final para o solicitante original (A)

        # 2b. Se NÃO HÁ conflito local
        else:
            print(f"[{self.node_id.upper()}] (Conselheiro) Análise local sem conflito. Decisão: {classification}.")
            # Retorna a classificação local (INTRUSION ou NORMAL)
            # Nota: 'classification' já será '0', '1', '2' etc.
            # Precisamos mapear isso para "INTRUSION" ou "NORMAL"
            decision = "NORMAL" if classification == '0' else "INTRUSION"

            if self._poisoning_active():
                decision = self._poison(decision) #envenamento de decisão

            print(f"[{self.node_id.upper()}] (Conselheiro) Decisão final enviada: {decision}")

            return decision

    def check_traffic_and_act(self, sample_data_array, ground_truth):
        """
        Processa uma amostra de tráfego local e verifica por conflito.
        """
        # Exibe as primeiras 5 features da amostra
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

            # Busca outros pares para consultar (exclui a si mesmo)
            other_peers = self.peer_manager.get_other_peers()

            if not other_peers:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito detectado, mas não há outros pares para consultar. Usando decisão local padrão.")
                return

            # O cliente agora selecionará um par aleatório da lista
            # Passa o ground_truth para o cliente para fins de logging
            counsel_response = self.client.request_counsel(
                sample_data_array,
                other_peers,
                ground_truth
            )

            if counsel_response and counsel_response['decision'] == 'INTRUSION':
                print(
                    f"[{self.node_id.upper()}] Ação: Decisão Final: INTRUSION (Confirmado por {counsel_response['counselor_id']}).")
            else:
                print(
                    f"[{self.node_id.upper()}] Ação: Conflito resolvido ou sem conselho externo definitivo. Usando decisão local padrão.")
        else:
            # Se não há conflito, usamos a decisão local
            final_decision = "INTRUSION" if classification != '0' else "NORMAL"
            print(
                f"[{self.node_id.upper()}] Classificação Local: Sem conflito detectado. Decisão Final: {final_decision}.")