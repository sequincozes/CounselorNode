import sys
import time
import numpy as np
import json

# Importing layers
from infrastructure.config_manager import ConfigManager
from infrastructure.networking import CounselorServer, CounselorClient
from core.classifier_engine import ClassifierEngine


class CounselorNode:
    """The main class representing the IDS (Detector) in the Counselors Network."""

    def __init__(self, node_id):
        self.node_id = node_id

        # Component 1: Configuration and Peers
        self.peer_manager = ConfigManager(node_id)
        local_info = self.peer_manager.get_local_info()
        self.host = local_info['ip']
        self.port = local_info['port']
        self.source_type = local_info['source_type']

        # Component 2: ML Engine (IDS Intelligence)
        self.engine = ClassifierEngine(self.peer_manager.get_ml_config())  # Initializes DCS training

        # Component 3 & 4: Network (The server receives the real classification function)
        self.client = CounselorClient(node_id, self.peer_manager)
        self.server = CounselorServer(
            self.host,
            self.port,
            self.node_id,
            self.source_type,
            self._execute_counseling_logic  # Passes the node's real logic as callback
        )

        print(f"--- {self.node_id.upper()} STARTED ---")
        print(f"Address: {self.host}:{self.port}")
        print(f"Specialty: {self.source_type}")
        print("-" * 30)

    def start(self):
        """Starts the server and keeps the node active."""
        self.server.start_listening()

    def _execute_counseling_logic(self, sample_data_array):
        """
        Callback function executed when this node receives a counseling request.
        """
        return self.engine.counseling_logic(sample_data_array)

    def check_traffic_and_act(self, sample_data_array):
        """
        Processes a sample using the local DCS and checks for conflict.
        """
        # Displays the first 5 features of the sample
        print(f"\n[{self.node_id.upper()}] Analyzing sample (first 5 features): {sample_data_array[:5]}...")

        # 1. Classifies and checks for conflict using the DCS engine
        results = self.engine.classify_and_check_conflict(sample_data_array)

        classification = results['classification']
        conflict = results['conflict']
        decisions = results['decisions']
        cluster_id = results.get('cluster_id', 'N/A')

        print(f"[{self.node_id.upper()}] Local DCS Result (Cluster {cluster_id}): {classification}")
        print(f"[{self.node_id.upper()}] Local Decisions (Classes): {decisions}")

        # 2. Counseling Network Trigger Logic
        if conflict:
            print(f"[{self.node_id.upper()}] Alert: CLASSIFIER CONFLICT DETECTED! Reaching out to Counselors Network.")

            # Chooses the specialist: Here, hardcoded for 'log_aplicacao' (Node B)
            required_type = 'log_aplicacao'

            counsel_response = self.client.request_counsel(sample_data_array, required_type)

            if counsel_response and counsel_response['decision'] == 'INTRUSION':
                print(
                    f"[{self.node_id.upper()}] Action: Final Decision: INTRUSION (Confirmed by {counsel_response['counselor_id']}).")
                # EXTENSION POINT: Implement Incremental Learning
            else:
                print(
                    f"[{self.node_id.upper()}] Action: Conflict resolved or no definitive external counsel. Using default local decision.")
        else:
            print(
                f"[{self.node_id.upper()}] Local Classification: No conflict detected. Final Decision: {classification}.")