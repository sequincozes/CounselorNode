import random
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

    def __init__(self, ml_config, node_dataset=None):
        self.config = ml_config
        self.node_dataset = node_dataset
        self.n_clusters = self.config.get('clustering_n_clusters', 5)
        # Threshold is 5% (0.05) for F1-score equivalence
        self.f1_threshold = self.config.get('f1_threshold', 0.05)
        self.cluster_classifiers = {}  # {cluster_id: {'best_models': [model1, model2], 'max_f1': 0.95}}
        self.scaler = None
        self.kmeans = None
        self.X_test = None
        self.test_size = 0.2 #test size alterado para Random para testes

        self._load_data()
        self._train_dcs_model()

    def _load_custom_data(self,data_source):
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
        
        return X, y
         
    def _load_data(self):
        """
        Loads the dataset based on the 'target_dataset' configuration.
        If 'auto' or train_source is None, uses the Breast Cancer dataset from sklearn.
        If a path is provided, it should load training and test data from that path.
        """
        train_source = self.node_dataset
        test_source = self.config.get('target_dataset', None)

        if test_source == 'auto' or train_source is None:
            # Option 1: Use sklearn's load_breast_cancer dataset as the standard dataset
            print("INFO: 'auto' source detected. Loading standard Scikit-learn 'Breast Cancer' dataset...")

            data = load_breast_cancer()
            X = data.data
            y = data.target

            print(
                f"INFO: Using fixed dataset parameters: {X.shape[0]} samples, {X.shape[1]} features, {np.unique(y).size} classes.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42) 
        
        else:
            # Opção para carregar um dataset externo
            print(f"INFO: Loading training data from specified source: {train_source}...")
            X_train, y_train = self._load_custom_data(train_source)

            print(f"INFO: Loading test from specified source: {test_source}...")
            X_test, y_test = self._load_custom_data(test_source)

        # Check if data was loaded successfully
        if X_train is None or y_train is None:
            print("ERROR: Train Data could not be loaded or generated.")
            sys.exit(1)

        if X_test is None or y_test is None:
            print("ERROR: Test Data could not be loaded or generated.")
            sys.exit(1)


        self.scaler = StandardScaler()

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_test = y_test

        print(f"DATASET: {self.X_train.shape[0]} samples for training.\n Test_size={self.test_size:.2f}")

    def _apply_clustering(self):
        """Applies the K-Means algorithm to the training data."""
        print(f"DCS: Applying K-Means with K={self.n_clusters}...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.kmeans.fit(self.X_train)
        self.labels = self.kmeans.labels_

        # Structures data by cluster
        clustered_data = {}
        for cluster_id in range(self.n_clusters):
            idx = np.where(self.labels == cluster_id)[0]
            clustered_data[cluster_id] = {
                'X': self.X_train[idx],
                'y': self.y_train[idx]
            }
        return clustered_data

    def _train_and_evaluate(self, cluster_data):
        """
        Trains and evaluates all candidate classifiers for a given cluster,
        using F1-Score as the metric.
        """
        results = {}
        X_cluster, y_cluster = cluster_data['X'], cluster_data['y']

        if len(X_cluster) == 0:
            return results

        for name in self.config['classifiers']:
            try:
                ClassifierClass = CLASSIFIER_MAP[name]
                model = ClassifierClass(random_state=42) if name not in ['GaussianNB',
                                                                         'KNeighborsClassifier'] else ClassifierClass()

                model.fit(X_cluster, y_cluster)

                # Evaluates on the cluster's own sub-dataset
                y_pred = model.predict(X_cluster)
                # Use 'weighted' average since classes are likely imbalanced after clustering
                f1 = f1_score(y_cluster, y_pred, average='weighted', zero_division=0)

                results[name] = {'model': model, 'f1_score': f1}
            except Exception as e:
                print(f"WARNING: Classifier {name} failed for cluster (Error: {e}).")

        return results

    def _train_dcs_model(self):
        """
        Implements the main DCS logic: K-Means, Evaluation, Selection.
        Prints the F1-Score of all classifiers for each cluster.
        """
        clustered_data = self._apply_clustering()

        print("DCS: Training and evaluating classifiers per cluster...")

        for cluster_id, data in clustered_data.items():
            evaluation_results = self._train_and_evaluate(data)

            print(f"\n--- Cluster {cluster_id} (N={len(data['X'])}) ---")

            if not evaluation_results:
                print(f"Cluster {cluster_id}: No data or classifiers failed.")
                continue

            # Print detailed F1-Scores
            scores = {}
            for name, res in evaluation_results.items():
                scores[name] = res['f1_score']
                print(f"  > {name}: F1-Score = {res['f1_score']:.4f}")

            max_f1 = max(scores.values())

            # Selects all classifiers that are within the 5% threshold (DCS Selection Logic)
            best_classifiers = []

            for name, res in evaluation_results.items():
                if res['f1_score'] >= (max_f1 - self.f1_threshold):
                    best_classifiers.append({
                        'name': name,
                        'model': res['model'],
                        'f1': res['f1_score']
                    })

            self.cluster_classifiers[cluster_id] = {
                'best_models': best_classifiers,
                'max_f1': max_f1
            }

            model_names = [c['name'] for c in best_classifiers]
            print(f"DCS SELECTION: Max F1={max_f1:.4f}. Threshold={self.f1_threshold:.4f}.")
            print(f"DCS SELECTION: Selected Models: {', '.join(model_names)}")

    def classify_and_check_conflict(self, sample_data):
        """
        Classifies a sample and checks for conflict among the best classifiers.
        Returns the classification result, conflict status, and the individual decisions.
        """
        # 1. Preprocess and find the cluster
        sample_scaled = self.scaler.transform(sample_data.reshape(1, -1))
        if self.kmeans is None:
            return {"classification": "NORMAL", "conflict": False, "decisions": ["Engine not initialized."]}

        cluster_id = self.kmeans.predict(sample_scaled)[0]

        if cluster_id not in self.cluster_classifiers:
            return {"classification": "UNKNOWN", "conflict": False, "decisions": ["Unmapped cluster."]}

        dcs_data = self.cluster_classifiers[cluster_id]

        # 2. Classify with selected models and record decisions
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
            final_class = unique_decisions[0] if len(unique_decisions) > 0 else "Nenhum classificador foi capaz de classificar." ## Checagem provavelmente desnecessária

        return {
            "classification": final_class,
            "conflict": conflict,
            "decisions": decisions,
            "cluster_id": int(cluster_id)
        }

    def counseling_logic(self, sample_data):
        """
        High-confidence classification logic used when the node acts as Counselor.
        Uses the model with the ABSOLUTE HIGHEST F1-Score in the cluster.
        """
        # final_prediction = best_model_info['model'].predict(sample_scaled)[0]
        final_prediction = self.classify_and_check_conflict(sample_data)
        print("##### Amostra que deu conflito no amiguinho: ")
        print(sample_data)
        print("##### Minha decisão sobre a amostra que deu conflito no amiguinho: ")
        print(final_prediction)
        return str(final_prediction)