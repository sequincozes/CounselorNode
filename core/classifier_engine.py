import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=FutureWarning)

CLASSIFIER_MAP = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "SVC": SVC,
}


class ClassifierEngine:
    """
    MODELOS GLOBAIS + AVALIAÇÃO POR CLUSTER + OUTLIER (OOD) por distância ao centróide.

    - Entrada de inferência: RAW (engine escala internamente).
    - Mantém X_train_raw/X_eval_raw/X_final_test_raw e versões escaladas.
    - Committee por cluster escolhido com base no EVAL do cluster.
    - Guarda também o F1 de cada classificador por cluster (para logging).
    """

    def __init__(self, ml_config):
        self.config = ml_config

        self.n_clusters = int(self.config.get("clustering_n_clusters", 5))
        self.f1_threshold = float(self.config.get("f1_threshold", 0.05))
        self.f1_min_required = float(self.config.get("f1_min_required", 0.80))  # 0..1

        self.random_state = int(self.config.get("random_state", 42))
        self.use_stratify = bool(self.config.get("stratify", True))

        self.eval_size = float(self.config.get("eval_size", 0.30))
        self.min_cluster_eval_samples = int(self.config.get("min_cluster_eval_samples", 10))

        self.outlier_enabled = bool(self.config.get("outlier_enabled", True))
        self.outlier_percentile = float(self.config.get("outlier_percentile", 97.0))

        self.train_eval_source = self.config.get(
            "train_eval_dataset_source",
            self.config.get("training_dataset_source", "auto")
        )
        self.final_test_source = self.config.get("final_test_dataset_source", None)
        self.target_column = self.config.get("target_column", None)

        # RAW
        self.X_train_raw = None
        self.X_eval_raw = None
        self.X_final_test_raw = None

        # SCALED
        self.X_train = None
        self.X_eval = None
        self.X_final_test = None

        # Labels
        self.y_train = None
        self.y_eval = None
        self.y_final_test = None

        # Models & clustering
        self.scaler = None
        self.kmeans = None
        self.clusters_train = None
        self.clusters_eval = None

        # Global pool
        self.global_models = {}  # {name: model}

        # Per cluster
        # cluster_id -> dict com best_models/max_f1/.../f1_by_classifier
        self.cluster_classifiers = {}

        # Outlier thresholds per cluster
        self.cluster_outlier_thresholds = {}
        self.cluster_outlier_stats = {}

        # Build
        self._load_train_eval_data()
        self._fit_scaler_and_cluster()
        self._fit_outlier_thresholds_from_train()
        self._train_global_models()
        self._select_committee_per_cluster()
        self._load_final_test_data()

    # ---------- utils ----------
    def _resolve_path(self, filename):
        if filename == "auto":
            return "auto"
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, "..", filename)

    def _load_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"[ENGINE] ERROR: Could not read dataset from {filepath}. Error: {e}")
            sys.exit(1)

        target_col = self.target_column
        if target_col is None:
            print("[ENGINE] WARNING: No 'target_column' specified. Assuming last column is the target.")
            target_col = df.columns[-1]

        if target_col not in df.columns:
            print(f"[ENGINE] ERROR: Target column '{target_col}' not found in dataset.")
            sys.exit(1)

        y = df[target_col].values
        X_df = df.drop(columns=[target_col])

        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X_df.values

        return X, y

    def _safe_split(self, X, y, test_size):
        strat = y if (self.use_stratify and len(np.unique(y)) > 1) else None
        try:
            return train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=strat
            )
        except ValueError:
            return train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=None
            )

    # ---------- load A ----------
    def _load_train_eval_data(self):
        src = self._resolve_path(self.train_eval_source)
        print(f"[ENGINE] Loading TRAIN+EVAL source: {src}")

        if src == "auto":
            data = load_breast_cancer()
            X = data.data
            y = data.target
        else:
            print(f"[ENGINE] INFO: Loading TRAIN+EVAL data from: {src}")
            X, y = self._load_csv(src)
            print(f"[ENGINE] INFO: TRAIN+EVAL loaded. {X.shape[0]} samples, {X.shape[1]} features.")

        X_train, X_eval, y_train, y_eval = self._safe_split(X, y, test_size=self.eval_size)

        self.X_train_raw = X_train
        self.X_eval_raw = X_eval
        self.y_train = y_train
        self.y_eval = y_eval

        print(f"[ENGINE] SPLIT A: TRAIN={self.X_train_raw.shape[0]} | EVAL={self.X_eval_raw.shape[0]}")

    # ---------- scaler + kmeans ----------
    def _fit_scaler_and_cluster(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train_raw)
        self.X_eval = self.scaler.transform(self.X_eval_raw)

        print(f"[ENGINE] Fitting K-Means on TRAIN (K={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        self.kmeans.fit(self.X_train)

        self.clusters_train = self.kmeans.predict(self.X_train)
        self.clusters_eval = self.kmeans.predict(self.X_eval)

        print("[ENGINE] K-Means fit complete. Clusters assigned for TRAIN and EVAL.")

    # ---------- outlier thresholds ----------
    def _fit_outlier_thresholds_from_train(self):
        self.cluster_outlier_thresholds = {}
        self.cluster_outlier_stats = {}

        if not self.outlier_enabled:
            for c in range(self.n_clusters):
                self.cluster_outlier_thresholds[c] = float("inf")
            print("[ENGINE] Outlier detection disabled.")
            return

        centers = self.kmeans.cluster_centers_
        p = float(self.outlier_percentile)

        for c in range(self.n_clusters):
            idx = np.where(self.clusters_train == c)[0]
            if len(idx) == 0:
                self.cluster_outlier_thresholds[c] = float("inf")
                self.cluster_outlier_stats[c] = {"p": p, "mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
                continue

            Xc = self.X_train[idx]
            centroid = centers[c]
            dists = np.linalg.norm(Xc - centroid, axis=1)

            thr = float(np.percentile(dists, p))
            self.cluster_outlier_thresholds[c] = thr
            self.cluster_outlier_stats[c] = {
                "p": p,
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "max": float(np.max(dists)),
                "n": int(len(dists)),
            }

        print(f"[ENGINE] Outlier thresholds fit from TRAIN using percentile p={p:.1f}.")

    # ---------- instantiate ----------
    def _instantiate_classifier(self, name: str):
        cls = CLASSIFIER_MAP.get(name)
        if not cls:
            return None
        try:
            if name in ["GaussianNB", "KNeighborsClassifier"]:
                return cls()
            return cls(random_state=self.random_state)
        except TypeError:
            try:
                return cls()
            except Exception:
                return None

    # ---------- train global ----------
    def _train_global_models(self):
        self.global_models = {}
        print("[ENGINE] Training GLOBAL models on full TRAIN (excluding EVAL)...")

        if len(np.unique(self.y_train)) < 2:
            print("[ENGINE] WARNING: TRAIN has only one class. Some classifiers may fail.")

        for name in self.config.get("classifiers", []):
            model = self._instantiate_classifier(name)
            if model is None:
                continue
            try:
                if len(np.unique(self.y_train)) < 2 and name in ["SVC"]:
                    continue
                model.fit(self.X_train, self.y_train)
                self.global_models[name] = model
                print(f"[ENGINE] GLOBAL model trained: {name}")
            except Exception as e:
                print(f"[ENGINE] WARN: Failed to train GLOBAL model {name}: {repr(e)}")

        if not self.global_models:
            print("[ENGINE] ERROR: No global models trained.")
            sys.exit(1)

    # ---------- committee per cluster + store f1_by_classifier ----------
    def _select_committee_per_cluster(self):
        print("[ENGINE] Selecting committee per cluster using EVAL (global models)...")
        self.cluster_classifiers = {}

        for cluster_id in range(self.n_clusters):
            tr_idx = np.where(self.clusters_train == cluster_id)[0]
            ev_idx = np.where(self.clusters_eval == cluster_id)[0]

            X_ev_c = self.X_eval[ev_idx]
            y_ev_c = self.y_eval[ev_idx]

            if len(ev_idx) < self.min_cluster_eval_samples:
                self.cluster_classifiers[cluster_id] = {
                    "best_models": [],
                    "max_f1": -1.0,
                    "below_min_f1": True,
                    "reason": "EVAL_TOO_SMALL",
                    "n_train": int(len(tr_idx)),
                    "n_eval": int(len(ev_idx)),
                    "f1_by_classifier": {}
                }
                print(f"[ENGINE] Cluster {cluster_id}: NOT_TRUSTED | EVAL_TOO_SMALL | n_train={len(tr_idx)} n_eval={len(ev_idx)}")
                continue

            models_f1 = []
            max_f1 = -1.0

            for name, model in self.global_models.items():
                try:
                    preds = model.predict(X_ev_c)
                    f1 = f1_score(y_ev_c, preds, average="weighted", zero_division=0)
                    models_f1.append({"name": name, "model": model, "f1": float(f1)})
                    if f1 > max_f1:
                        max_f1 = float(f1)
                except Exception as e:
                    print(f"[ENGINE] WARN: eval failed for {name} on cluster {cluster_id}: {repr(e)}")

            f1_by_classifier = {m["name"]: float(m["f1"]) for m in models_f1}

            if not models_f1 or max_f1 < 0:
                self.cluster_classifiers[cluster_id] = {
                    "best_models": [],
                    "max_f1": float(max_f1),
                    "below_min_f1": True,
                    "reason": "NO_MODEL_EVALUATED",
                    "n_train": int(len(tr_idx)),
                    "n_eval": int(len(ev_idx)),
                    "f1_by_classifier": f1_by_classifier
                }
                continue

            below_min = (max_f1 < self.f1_min_required)

            cutoff = max_f1 - self.f1_threshold
            selected = [m for m in models_f1 if m["f1"] >= cutoff]

            if not selected:
                below_min = True
                reason_final = "NO_SELECTED_MODELS"
            else:
                reason_final = "OK" if not below_min else "MAX_F1_BELOW_MIN"

            self.cluster_classifiers[cluster_id] = {
                "best_models": selected,
                "max_f1": float(max_f1),
                "below_min_f1": bool(below_min),
                "reason": reason_final,
                "n_train": int(len(tr_idx)),
                "n_eval": int(len(ev_idx)),
                "f1_by_classifier": f1_by_classifier
            }

            print(
                f"[ENGINE] Cluster {cluster_id}: n_train={len(tr_idx)} n_eval={len(ev_idx)} | "
                f"max_f1(eval)={max_f1:.4f} | selected={len(selected)} | below_min={below_min}"
            )

        print("[ENGINE] Committee selection complete.")

    # ---------- load B RAW ----------
    def _load_final_test_data(self):
        if not self.final_test_source:
            print("[ENGINE] Final test source not provided. Skipping CSV B.")
            return

        src = self._resolve_path(self.final_test_source)
        print(f"[ENGINE] Loading FINAL TEST source (CSV B): {src}")

        if src == "auto":
            data = load_breast_cancer()
            X = data.data
            y = data.target
        else:
            X, y = self._load_csv(src)
            print(f"[ENGINE] INFO: FINAL TEST loaded. {X.shape[0]} samples, {X.shape[1]} features.")

        if X.shape[1] != self.X_train_raw.shape[1]:
            print(f"[ENGINE] ERROR: FINAL TEST has {X.shape[1]} features, TRAIN has {self.X_train_raw.shape[1]}.")
            sys.exit(1)

        self.X_final_test_raw = X
        self.y_final_test = y
        self.X_final_test = self.scaler.transform(X)  # opcional

        print(f"[ENGINE] FINAL TEST ready: {self.X_final_test_raw.shape[0]} samples.")

    # ---------- inference ----------
    def classify_and_check_conflict(self, sample_raw):
        if self.scaler is None or self.kmeans is None:
            return {"classification": "UNKNOWN", "conflict": True, "decisions": ["Engine not ready"], "cluster_id": -1}

        sample_scaled = self.scaler.transform(sample_raw.reshape(1, -1))
        cluster_id = int(self.kmeans.predict(sample_scaled)[0])

        if self.outlier_enabled:
            centroid = self.kmeans.cluster_centers_[cluster_id]
            dist = float(np.linalg.norm(sample_scaled[0] - centroid))
            thr = float(self.cluster_outlier_thresholds.get(cluster_id, float("inf")))
            if dist > thr:
                return {
                    "classification": "CONFLICT_DETECTED",
                    "conflict": True,
                    "decisions": [f"OOD_OUTLIER(dist={dist:.4f} > thr_p{self.outlier_percentile:.1f}={thr:.4f})"],
                    "cluster_id": cluster_id,
                }

        if cluster_id not in self.cluster_classifiers:
            return {"classification": "UNKNOWN", "conflict": True, "decisions": ["Cluster not mapped"], "cluster_id": cluster_id}

        dcs = self.cluster_classifiers[cluster_id]

        if dcs.get("below_min_f1", False):
            reason = dcs.get("reason", "UNTRUSTED_CLUSTER")
            max_f1 = dcs.get("max_f1", -1.0)
            return {
                "classification": "CONFLICT_DETECTED",
                "conflict": True,
                "decisions": [f"UNTRUSTED_CLUSTER({reason})", f"min_f1={self.f1_min_required:.3f}", f"max_f1={max_f1:.3f}"],
                "cluster_id": cluster_id,
            }

        models = dcs.get("best_models", [])
        if not models:
            return {"classification": "CONFLICT_DETECTED", "conflict": True, "decisions": ["No selected models"], "cluster_id": cluster_id}

        decisions = []
        for info in models:
            try:
                pred = info["model"].predict(sample_scaled)[0]
                decisions.append(str(pred))
            except Exception:
                decisions.append("ERROR")

        unique = np.unique(decisions)
        conflict = len(unique) > 1
        final_class = "CONFLICT_DETECTED" if conflict else (unique[0] if len(unique) else "UNKNOWN")

        return {"classification": final_class, "conflict": bool(conflict), "decisions": decisions, "cluster_id": cluster_id}

    def counseling_logic(self, sample_raw):
        sample_scaled = self.scaler.transform(sample_raw.reshape(1, -1))
        cluster_id = int(self.kmeans.predict(sample_scaled)[0])

        if cluster_id not in self.cluster_classifiers:
            return "UNKNOWN"

        dcs = self.cluster_classifiers[cluster_id]
        if dcs.get("below_min_f1", False):
            return "UNKNOWN"

        models = dcs.get("best_models", [])
        if not models:
            return "UNKNOWN"

        best = max(models, key=lambda x: x.get("f1", -1.0), default=None)
        if best is None:
            return "UNKNOWN"

        try:
            pred = best["model"].predict(sample_scaled)[0]
            return str(pred)
        except Exception:
            return "UNKNOWN"

    # ---------- learning ----------
    def add_training_sample_raw(self, sample_raw, label, retrain=False):
        sample_raw = np.asarray(sample_raw, dtype=float).reshape(-1)

        if self.X_train_raw is None:
            self.X_train_raw = sample_raw.reshape(1, -1)
            self.y_train = np.array([label])
        else:
            self.X_train_raw = np.vstack([self.X_train_raw, sample_raw.reshape(1, -1)])
            self.y_train = np.append(self.y_train, label)

        if retrain:
            self.rebuild()

    def rebuild(self):
        # refit scaler
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train_raw)
        self.X_eval = self.scaler.transform(self.X_eval_raw)

        # kmeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        self.kmeans.fit(self.X_train)

        self.clusters_train = self.kmeans.predict(self.X_train)
        self.clusters_eval = self.kmeans.predict(self.X_eval)

        # outlier
        self._fit_outlier_thresholds_from_train()

        # models + committee
        self._train_global_models()
        self._select_committee_per_cluster()

    # ---------- snapshot rows (inclui f1_by_classifier) ----------
    def get_cluster_f1_snapshot_rows(self):
        rows = []
        for cluster_id in range(self.n_clusters):
            d = self.cluster_classifiers.get(cluster_id, {})

            best_models = d.get("best_models", []) or []
            selected_names = []
            for m in best_models:
                if isinstance(m, dict) and "name" in m:
                    selected_names.append(str(m["name"]))

            f1_map = d.get("f1_by_classifier", {}) or {}
            f1_serialized = "|".join([f"{k}:{float(v):.6f}" for k, v in f1_map.items()])

            rows.append({
                "cluster_id": int(cluster_id),
                "max_f1": float(d.get("max_f1", -1.0)),
                "selected_models": "|".join(selected_names),
                "below_min_f1": bool(d.get("below_min_f1", True)),
                "reason": str(d.get("reason", "N/A")),
                "n_train": int(d.get("n_train", -1)),
                "n_eval": int(d.get("n_eval", -1)),
                "f1_by_classifier": f1_serialized
            })

        return rows
