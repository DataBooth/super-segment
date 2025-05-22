import datetime
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from streamlit import spinner


def compute_data_hash(df: pd.DataFrame) -> str:
    """Returns a hex digest hash for the entire DataFrame contents."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    m = hashlib.sha256()
    m.update(row_hashes.tobytes())
    return m.hexdigest()


def get_or_train_model(
    data,
    model_path="data/model.pkl",
    n_clusters=4,
    use_age_cohort=False,
    force_retrain=False,
):
    if Path(model_path).exists() and not force_retrain:
        model, metadata, fit_stats = load_model_with_metadata(model_path)
        current_hash = compute_data_hash(data)
        if (
            metadata["n_member"] == len(data)
            and metadata.get("n_clusters", None) == n_clusters
            and metadata.get("use_age_cohort", False) == use_age_cohort
            and metadata.get("data_hash", None) == current_hash
        ):
            logger.info(f"Loaded cached model (trained {metadata['train_time']})")
            return model, metadata, fit_stats
        else:
            logger.info("Cached model metadata mismatch; retraining.")
    # Train and save new model
    model = SuperannuationSegmentationModel(n_clusters=n_clusters)
    fit_stats = model.train(data)
    save_model_with_metadata(model, data, model_path, fit_stats)
    _, metadata, _ = load_model_with_metadata(model_path)
    return model, metadata, fit_stats


def save_model_with_metadata(model, data, model_path, fit_stats=None):
    metadata = {
        "n_member": len(data),
        "train_time": datetime.datetime.now().isoformat(),
        "data_hash": compute_data_hash(data),
        "n_clusters": getattr(model, "n_clusters", None),
    }
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata, "fit_stats": fit_stats}, f)


def load_model_with_metadata(model_path):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["metadata"], obj.get("fit_stats")


class SuperannuationSegmentationModel:
    """
    Trains and predicts member segments using clustering (KMeans) on superannuation data.
    """

    def __init__(self, n_clusters=4):
        self.numeric_features = [
            "age",
            "balance",
            "num_accounts",
            "last_login_days",
            "satisfaction_score",
            "logins_per_month",
        ]
        self.categorical_features = [
            "profession",
            "phase",
            "gender",
            "region",
            "risk_profile",
        ]
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = None
        self.encoder = None
        self.is_trained = False
        self.feature_columns = None

    def preprocess(self, df: pd.DataFrame):
        X_num = df[self.numeric_features].copy()
        X_cat = df[self.categorical_features].copy()
        if self.scaler is None:
            self.scaler = StandardScaler().fit(X_num)
        X_num_scaled = self.scaler.transform(X_num)
        if self.encoder is None:
            self.encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            ).fit(X_cat)
            self.feature_columns = self.numeric_features + list(
                self.encoder.get_feature_names_out(self.categorical_features)
            )
        X_cat_encoded = self.encoder.transform(X_cat)
        X = np.hstack([X_num_scaled, X_cat_encoded])
        return X

    def train(self, df: pd.DataFrame) -> dict:
        with spinner(
            f"Training segmentation model on {len(df):,} members. Please wait..."
        ):
            X = self.preprocess(df)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            cluster_labels = self.kmeans.fit_predict(X)
            df["segment"] = cluster_labels
            sil = silhouette_score(X, cluster_labels)
            self.is_trained = True
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            return {
                "silhouette": sil,
                "cluster_sizes": cluster_sizes,
            }

    def predict_segment(self, input_data: dict) -> int:
        df_input = pd.DataFrame([input_data])
        X = self.preprocess(df_input)
        segment = int(self.kmeans.predict(X)[0])
        logger.info(f"Predicted segment: {segment}")
        return segment

    def add_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.preprocess(df)
        df = df.copy()
        df["segment"] = self.kmeans.predict(X)
        return df

    def visualise_clusters(self, df: pd.DataFrame):
        X = self.preprocess(df)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_vis = df.copy()
        df_vis["PCA1"] = X_pca[:, 0]
        df_vis["PCA2"] = X_pca[:, 1]
        fig = px.scatter(
            df_vis,
            x="PCA1",
            y="PCA2",
            color="segment",
            symbol="segment",
            title="Member Segments (PCA projection)",
            hover_data=self.numeric_features + self.categorical_features,
        )
        return fig

    def cohort_sensitivity_analysis(
        self,
        df: pd.DataFrame,
        base_cohorts=None,
        boundary_shifts=(-2, -1, 0, 1, 2),
        min_cluster_size=10,
        n_clusters=None,
    ):
        if base_cohorts is None:
            base_cohorts = [(18, 30), (30, 40), (40, 50), (50, 65), (65, 120)]
        if n_clusters is None:
            n_clusters = self.n_clusters
        results = []
        for shift in boundary_shifts:
            cohorts = [(low + shift, high + shift) for (low, high) in base_cohorts]
            for i, (low, high) in enumerate(cohorts):
                mask = (df["age"] >= low) & (df["age"] < high)
                df_cohort = df[mask]
                if len(df_cohort) < n_clusters or len(df_cohort) < min_cluster_size:
                    results.append(
                        {
                            "shift": shift,
                            "cohort": f"{low}-{high}",
                            "n_members": len(df_cohort),
                            "silhouette": np.nan,
                            "cluster_sizes": None,
                        }
                    )
                    continue
                try:
                    X = self.preprocess(df_cohort)
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    labels = kmeans.fit_predict(X)
                    sil = silhouette_score(X, labels)
                    cluster_sizes = (
                        pd.Series(labels).value_counts().sort_index().tolist()
                    )
                    results.append(
                        {
                            "shift": shift,
                            "cohort": f"{low}-{high}",
                            "n_members": len(df_cohort),
                            "silhouette": sil,
                            "cluster_sizes": cluster_sizes,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Clustering failed for cohort {low}-{high} (shift {shift}): {e}"
                    )
                    results.append(
                        {
                            "shift": shift,
                            "cohort": f"{low}-{high}",
                            "n_members": len(df_cohort),
                            "silhouette": np.nan,
                            "cluster_sizes": None,
                        }
                    )
        results_df = pd.DataFrame(results)
        return results_df
