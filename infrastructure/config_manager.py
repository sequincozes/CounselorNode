import json
import sys
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'peer_config.json')


class ConfigManager:
    """Manages loading and querying the list of P2P peers and ML configurations."""

    def __init__(self, node_id):
        self.node_id = node_id
        self.config = self._load_config()

        if node_id not in self.config:
            raise ValueError(f"Node ID '{node_id}' not found in configuration.")

        self.local_info = self.config[node_id]
        self.peers = self.config.get('counselor_peers', [])
        # Load ML configuration
        self.ml_config = self.config.get('ml_config', {})

    def _load_config(self):
        """Loads the configuration JSON file."""
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found at: {CONFIG_PATH}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERROR: Invalid configuration file.")
            sys.exit(1)

    def get_local_info(self):
        """Returns the local node's IP/Port/Specialty information."""
        return self.local_info

    def get_ml_config(self):
        """Returns the Machine Learning configuration dictionary."""
        return self.ml_config

    def find_counselor(self, required_source_type):
        """Finds a peer with the required specialty."""
        return next((p for p in self.peers if p['source_type'] == required_source_type), None)