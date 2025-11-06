import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
import sys
import pandas as pd

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Maps string names to Scikit-learn classifier classes
CLASSIFIER_MAP = {
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
    'GaussianNB': GaussianNB,
    'SVC': SVC
}


class ClassifierEngine:
    """
    Manages the Dynamic Classifier Selection (DCS) mechanism for a single Counselor Node.
    It handles data loading, K-Means clustering, classifier training, and conflict detection.
    """

    def __init__(self, ml_config):
        self.config = ml_config
        self.n_clusters = self.config.get('clustering_n_clusters', 5)
        # Threshold is 5% (0.05) for F1-score equivalence
        self.f1_threshold = self.config.get('f1_threshold', 0.05)
        self.cluster_classifiers = {}  # {cluster_id: {'best_models': [...]}}
        self.kmeans = None
        self.scaler = None
        self.X_test = None  # Store test data for simulation
        self.y_test = None  # Store test labels for simulation (ground truth)

        # Start the training process upon initialization
        self._load_and_train()

    def _load_and_train(self):
        """Loads data, scales, clusters, and trains all classifiers for DCS."""
        print("Iniciando motor de classificação (Carregamento e Treinamento)...")
        try:
            # 1. Load Data
            # Using Breast Cancer dataset as a placeholder
            if self.config.get('training_dataset_source') == 'auto':
                data = load_breast_cancer()
                X, y = data.data, data.target
                # Ensure it matches config (example data might not)
                X = X[:, :self.config.get('n_features', 30)]
            else:
                # Load from CSV
                dataset_path = self.config.get('training_dataset_source')
                df = pd.read_csv(dataset_path)
                # Assuming the last column is the target
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values

            # 2. Scale and Split Data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )

            # 3. K-Means Clustering
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            train_clusters = self.kmeans.fit_predict(X_train)
            print(f"Treinamento K-Means concluído. {self.n_clusters} clusters definidos.")

            # 4. Train Classifiers for each Cluster (DCS)
            classifier_names = self.config.get('classifiers', ['DecisionTreeClassifier'])

            for cluster_id in range(self.n_clusters):
                # Get data points for this cluster
                cluster_indices = np.where(train_clusters == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue  # Skip empty clusters

                X_cluster, y_cluster = X_train[cluster_indices], y_train[cluster_indices]

                best_models_for_cluster = []
                max_f1 = -1.0

                for name in classifier_names:
                    if name not in CLASSIFIER_MAP:
                        continue

                    model = CLASSIFIER_MAP[name]()

                    # Ensure cluster has enough samples for all classes
                    if len(np.unique(y_cluster)) < 2:
                        # Cannot train or get F1 score
                        continue

                    model.fit(X_cluster, y_cluster)

                    # Evaluate on test set
                    test_clusters = self.kmeans.predict(self.X_test)
                    test_cluster_indices = np.where(test_clusters == cluster_id)[0]

                    if len(test_cluster_indices) == 0:
                        continue  # No test samples in this cluster

                    X_test_cluster, y_test_cluster = self.X_test[test_cluster_indices], self.y_test[
                        test_cluster_indices]

                    if len(np.unique(y_test_cluster)) < 2:
                        continue  # Cannot get F1 score if only one class in test set

                    y_pred = model.predict(X_test_cluster)
                    f1 = f1_score(y_test_cluster, y_pred, average='weighted', zero_division=0)

                    # Store model if its F1 is high enough
                    if f1 > (max_f1 - self.f1_threshold):
                        if f1 > max_f1:
                            max_f1 = f1

                        best_models_for_cluster.append({
                            'name': name,
                            'f1': f1,
                            'model': model
                        })

                # Filter only models that are "best" (within threshold)
                if best_models_for_cluster:
                    final_best_models = [m for m in best_models_for_cluster if m['f1'] >= (max_f1 - self.f1_threshold)]
                    self.cluster_classifiers[cluster_id] = {'best_models': final_best_models}
                    # print(f"Cluster {cluster_id}: {len(final_best_models)} melhores modelos (Max F1: {max_f1:.4f})")

            print("Motor de classificação pronto.")

        except FileNotFoundError:
            print(f"ERRO: Arquivo de dataset não encontrado em: {self.config.get('training_dataset_source')}")
            sys.exit(1)
        except Exception as e:
            print(f"ERRO durante o treinamento do Motor de ML: {e}")
            sys.exit(1)

    def classify_and_check_conflict(self, sample_data_array):
        """
        Performs Dynamic Classifier Selection (DCS) and checks for conflicts.
        Returns a dictionary with the result.
        """
        # 1. Preprocess the single sample
        try:
            sample_scaled = self.scaler.transform(sample_data_array.reshape(1, -1))
        except Exception as e:
            print(f"ERRO de Engine: Falha ao normalizar amostra: {e}")
            return {"classification": "UNKNOWN", "conflict": False, "decisions": [], "cluster_id": "N/A"}

        if self.kmeans is None:
            return {"classification": "UNKNOWN", "conflict": False, "decisions": [], "cluster_id": "N/A"}

        # 2. Find cluster and get best models
        cluster_id = self.kmeans.predict(sample_scaled)[0]

        if cluster_id not in self.cluster_classifiers:
            # No models trained for this cluster
            return {"classification": "UNKNOWN", "conflict": False, "decisions": [], "cluster_id": int(cluster_id)}

        dcs_data = self.cluster_classifiers[cluster_id]
        decisions = []

        for model_info in dcs_data['best_models']:
            prediction = model_info['model'].predict(sample_scaled)[0]
            decisions.append(str(prediction))  # Store decisions as strings

        # 3. Check for conflict (more than one unique decision)
        unique_decisions = np.unique(decisions)
        conflict = len(unique_decisions) > 1

        if conflict:
            final_class = "CONFLICT_DETECTED"
        else:
            # Takes the majority/unique decision
            final_class = unique_decisions[0] if len(unique_decisions) > 0 else "NORMAL"

        return {
            "classification": final_class,
            "conflict": conflict,
            "decisions": decisions,
            "cluster_id": int(cluster_id)
        }
