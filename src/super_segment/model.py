import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from loguru import logger
import numpy as np
import plotly.express as px
from streamlit import spinner, success


class SuperannuationSegmentationModel:
    """
    Trains and predicts member segments using clustering (KMeans) on superannuation data.
    """

    def __init__(self, n_clusters=4):
        # You can adjust features as needed
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
            "contrib_freq",
        ]
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = None
        self.encoder = None
        self.is_trained = False
        self.feature_columns = None

    def preprocess(self, df: pd.DataFrame):
        # Scale numeric, one-hot categorical, concatenate
        X_num = df[self.numeric_features].copy()
        X_cat = df[self.categorical_features].copy()

        # Fit/transform scaler and encoder if not already
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
            f"Training segmentation model on {len(df)} members. Please wait..."
        ):
            X = self.preprocess(df)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            cluster_labels = self.kmeans.fit_predict(X)
            df["segment"] = cluster_labels
            sil = silhouette_score(X, cluster_labels)
            self.is_trained = True
            # Save cluster sizes for reporting
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        success("Segmentation model trained!")
        return {
            "silhouette": sil,
            "cluster_sizes": cluster_sizes,
        }

    def predict_segment(self, input_data: dict) -> int:
        # Accepts a dict of member features, returns predicted segment
        df_input = pd.DataFrame([input_data])
        X = self.preprocess(df_input)
        segment = int(self.kmeans.predict(X)[0])
        logger.info(f"Predicted segment: {segment}")
        return segment

    def add_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assigns segments to the input DataFrame
        X = self.preprocess(df)
        df = df.copy()
        df["segment"] = self.kmeans.predict(X)
        return df

    def visualise_clusters(self, df: pd.DataFrame):
        # Visualise clusters using PCA for dimensionality reduction
        from sklearn.decomposition import PCA

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
