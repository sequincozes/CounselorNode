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
import os

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
        self.f1_threshold = self.config.get('f1_threshold', 0.15)
        self.cluster_classifiers = {}  # {cluster_id: {'best_models': [model1, model2], 'max_f1': 0.95}}
        self.scaler = None
        self.kmeans = None
        self.X_test = None
        self.y_test = None

        self._load_data()
        self._apply_clustering()
        self._train_dcs_model()

    def _load_data(self):
        """Loads and preprocesses the training data based on configuration."""
        data_source = self.config.get('training_dataset_source', 'auto')
        print(f"[ENGINE] Loading data source: {data_source}")

        if data_source == 'auto':
            data = load_breast_cancer()
            X = data.data
            y = data.target
        else:
            # Opção para carregar um dataset externo
            print(f"INFO: Loading data from specified source: {data_source}...")

            try:
                df = pd.read_csv(data_source)
            except Exception as e:
                print(f"ERROR: Could not read dataset from {data_source}. Error: {e}")
                sys.exit(1)

            # Verifica se existe uma coluna alvo
            target_col = self.config.get('target_column', None)
            if target_col is None:
                print("WARNING: No 'target_column' specified in configuration. Assuming last column is the target.")
                target_col = df.columns[-1]

            if target_col not in df.columns:
                print(f"ERROR: Target column '{target_col}' not found in dataset.")
                sys.exit(1)

            y = df[target_col].values
            X = df.drop(columns=[target_col]).values

            print(f"INFO: Dataset loaded from file. {X.shape[0]} samples, {X.shape[1]} features.")
            # -------------------------------------

            # Check if data was loaded successfully
        if X is None or y is None:
            print("ERROR: Data could not be loaded or generated.")
            sys.exit(1)

            # Split and Standardize data (common to both options)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"DATASET: {self.X_train.shape[0]} samples for training.")
        # else:
        #     try:
        #         df = pd.read_csv(source)
        #
        #         # Assume a coluna alvo é a última ('Label' ou 'Class')
        #         X = df.iloc[:, :-1].values
        #         y = df.iloc[:, -1].values
        #
        #     except (FileNotFoundError, IndexError, Exception) as e:
        #         print(f"ERRO: Falha ao carregar o dataset '{source}'. Usando fallback: {e}")
        #         data = make_classification(n_samples=500, n_features=20, n_informative=15, n_redundant=0, n_classes=3,
        #                                    n_clusters_per_class=1, random_state=42)
        #         X = data[0]
        #         y = data[1]
        #
        # # Train-test split (70% for training the DCS models, 30% for final test)
        # X_train, self.X_test, y_train, self.y_test = train_test_split(
        #     X, y, test_size=0.97, random_state=42, stratify=y
        # )
        #
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.7, random_state=42)
        #
        # # Standard Scaler (CRUCIAL for distance-based algorithms like KNN/SVC)
        # self.scaler = StandardScaler()
        # self.X_train_scaled = self.scaler.fit_transform(X_train)
        # self.y_train = y_train
        #
        # print(f"[ENGINE] Data loaded: {self.X_train_scaled.shape[0]} samples for training.")

    def _apply_clustering(self):
        """Applies K-Means clustering to the training data."""
        print(f"[ENGINE] Applying K-Means clustering (K={self.n_clusters})...")
        try:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            self.kmeans.fit(self.X_train)
            self.clusters = self.kmeans.predict(self.X_train)
            print("[ENGINE] K-Means clustering complete.")
        except ValueError as e:
            print(f"[ENGINE] ERRO ao aplicar K-Means. Verifique os dados: {e}")
            sys.exit(1)

    def _train_and_evaluate(self, cluster_data, cluster_labels):
        """Trains and evaluates all configured classifiers on a specific cluster's data."""
        best_f1 = -1
        models_f1 = []

        for name in self.config.get('classifiers', []):
            ClassifierClass = CLASSIFIER_MAP.get(name)
            if not ClassifierClass:
                continue

            try:
                model = ClassifierClass(random_state=42) if name not in ['GaussianNB',
                                                                         'KNeighborsClassifier'] else ClassifierClass()
                model.fit(cluster_data, cluster_labels)

                # Use a pequena porção de teste do próprio cluster para a pontuação F1
                # Nota: Idealmente, usaríamos um conjunto de validação separado, mas para DCS básico, isso serve.
                predictions = model.predict(cluster_data)

                # Weighted F1-Score é bom para classes desbalanceadas (comum em IDS)
                f1 = f1_score(cluster_labels, predictions, average='weighted', zero_division=0)

                models_f1.append({
                    'name': name,
                    'model': model,
                    'f1': f1
                })

                if f1 > best_f1:
                    best_f1 = f1

            except Exception as e:
                # print(f"Warning: Falha ao treinar {name} no cluster: {e}")
                continue

        return models_f1, best_f1

    def _train_dcs_model(self):
        """
        Trains the Dynamic Classifier Selection (DCS) mechanism.
        Selects a committee of models for each cluster based on F1-score.
        """
        print("[ENGINE] Starting DCS model training...")

        for cluster_id in range(self.n_clusters):
            # Isolate data for the current cluster
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            if len(cluster_indices) == 0:
                print(f"Warning: Cluster {cluster_id} está vazio. Pulando.")
                continue

            X_cluster = self.X_train[cluster_indices]
            y_cluster = self.y_train[cluster_indices]

            # Train and evaluate all models on the cluster
            models_f1, max_f1 = self._train_and_evaluate(X_cluster, y_cluster)

            # Select models whose F1 is close to the max F1 (within the threshold)
            best_models = [
                m for m in models_f1
                if m['f1'] >= (max_f1 - self.f1_threshold)
            ]

            if best_models:
                self.cluster_classifiers[cluster_id] = {
                    'best_models': best_models,
                    'max_f1': max_f1
                }
                # print(f"Cluster {cluster_id}: {len(best_models)} models selected. Max F1: {max_f1:.4f}")
            else:
                print(f"Cluster {cluster_id}: Nenhum modelo selecionado (max_f1: {max_f1:.4f}).")

        print("[ENGINE] DCS training complete.")

    def classify_and_check_conflict(self, sample_data):
        """
        Classifies a single sample using the DCS committee for its cluster
        and checks if the committee decision is unanimous (no conflict).
        """
        # 1. Scale the sample and predict the cluster
        sample_scaled = self.scaler.transform(sample_data.reshape(1, -1))

        if self.kmeans is None:
            return {"classification": "UNKNOWN", "conflict": True, "decisions": ["Engine not ready"], "cluster_id": -1}

        cluster_id = self.kmeans.predict(sample_scaled)[0]

        if cluster_id not in self.cluster_classifiers:
            # Não há classificadores treinados para este cluster
            return {"classification": "UNKNOWN", "conflict": True, "decisions": ["Cluster not mapped"],
                    "cluster_id": int(cluster_id)}

        dcs_data = self.cluster_classifiers[cluster_id]

        # 2. Classify with all selected models and record decisions
        decisions = []
        for classifier_info in dcs_data['best_models']:
            prediction = classifier_info['model'].predict(sample_scaled)[0]
            decisions.append(str(prediction))

        # 3. Check for conflict (more than one unique decision)
        unique_decisions = np.unique(decisions)
        conflict = len(unique_decisions) > 1

        if conflict:
            final_class = "CONFLICT_DETECTED"
        else:
            # Takes the majority/unique decision
            final_class = unique_decisions[0] if len(unique_decisions) > 0 else "UNKNOWN"

        return {
            "classification": final_class,
            "conflict": conflict,
            "decisions": decisions,
            "cluster_id": int(cluster_id)
        }

    def counseling_logic(self, sample_data):
        """
        Lógica de classificação de alta confiança usada quando o nó atua como Conselheiro
        ou quando é preciso dar uma resposta única (e.g., LOOP_CLOSED).
        Utiliza o modelo com o MAIOR F1-Score ABSOLUTO no cluster para dar um voto único.
        """
        # 1. Preprocessa e encontra o cluster
        sample_scaled = self.scaler.transform(sample_data.reshape(1, -1))
        if self.kmeans is None:
            return "UNKNOWN"  # Retorna UNKNOWN se o motor não estiver pronto

        cluster_id = self.kmeans.predict(sample_scaled)[0]

        if cluster_id not in self.cluster_classifiers:
            return "UNKNOWN"  # Retorna UNKNOWN se o cluster não estiver mapeado

        dcs_data = self.cluster_classifiers[cluster_id]

        # 2. Encontra o modelo ABSOLUTAMENTE melhor (maior F1-Score) dentro do conjunto selecionado

        # Encontra o modelo com maior F1-score absoluto no cluster
        best_model_info = max(dcs_data['best_models'], key=lambda x: x['f1'], default=None)

        if best_model_info is None:
            return "UNKNOWN"

        # 3. Classifica usando apenas o modelo mais confiável
        final_prediction = best_model_info['model'].predict(sample_scaled)[0]

        # Retorna a classe (ex: '0', '1', '2'...) para ser mapeada no node.py
        return str(final_prediction)
