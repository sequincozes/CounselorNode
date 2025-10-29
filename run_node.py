import sys
import time
import numpy as np
from core.node import CounselorNode


# Helper function to generate consistent test samples
def generate_test_sample(engine):
    """Generates a random sample from the ClassifierEngine's test set."""
    if engine.X_test is not None and len(engine.X_test) > 0:
        idx = np.random.randint(0, len(engine.X_test))
        # Returns the sample (already standardized)
        return engine.X_test[idx]

    return np.zeros(engine.config['n_features'])


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_node.py <node_id>")
        print("Example: python run_node.py node_a")
        sys.exit(1)

    node_id = sys.argv[1].lower()

    try:
        node = CounselorNode(node_id)
        node.start()

        # Traffic Simulation (Only for Node A, which requests counsel)
        if node_id == 'node_a':
            # Gives time for all setup (ML training can take a few seconds) and Node B to start
            time.sleep(10)

            print("\n" + "=" * 50)
            print("SIMULATION: NODE A (DCS) CHECKING TRAFFIC")
            print("=" * 50)

            while True:
                # Sample 1: Should be classified or generate conflict
                suspect_sample_data = generate_test_sample(node.engine)
                node.check_traffic_and_act(suspect_sample_data)

                time.sleep(0.1)  # Waits for P2P response


        # Main loop to keep the server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()