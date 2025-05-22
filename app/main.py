import streamlit as st
import plotly.express as px
import seaborn as sns
import duckdb

from super_segment.hydra_conf import load_hydra_conf

from super_segment.model import get_or_train_model
from super_segment.utils import display_markdown_file


class SuperSegmentApp:
    def __init__(self):
        self.config = load_hydra_conf()
        self.model = None

    def get_duckdb_data(self):
        parquet_path = self.config.data.output_file
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{parquet_path}'").df()
        con.close()
        return df

    def setup_session_state(self):
        if "data" not in st.session_state:
            st.session_state["data"] = self.get_duckdb_data()
        if (
            "model" not in st.session_state
            or "model_metadata" not in st.session_state
            or "fit_stats" not in st.session_state
        ):
            model, metadata, fit_stats = get_or_train_model(st.session_state["data"])
            st.session_state["model"] = model
            st.session_state["model_metadata"] = metadata
            st.session_state["fit_stats"] = fit_stats

    def sidebar_predict_form(self):
        st.header("Predict Segment for a Member")
        with st.form("predict_form"):
            age = st.slider("Age", 25, 70, 40)
            balance = st.number_input("Balance", 5000, 900000, 120000, step=1000)
            num_accounts = st.selectbox("Number of Accounts", [1, 2, 3, 4], index=0)
            last_login_days = st.slider("Days Since Last Login", 1, 180, 30)
            satisfaction_score = st.selectbox(
                "Satisfaction Score", [1, 2, 3, 4, 5], index=2
            )
            profession = st.selectbox(
                "Profession",
                [
                    "High School Teacher",
                    "Primary Teacher",
                    "Admin",
                    "TAFE Instructor",
                    "Principal",
                ],
            )
            phase = st.selectbox("Phase", ["Accumulation", "Retirement"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["NSW", "VIC", "QLD", "WA", "SA"])
            risk_profile = st.selectbox(
                "Risk Profile", ["Conservative", "Moderate", "Aggressive"]
            )
            logins_per_month = st.slider("Logins per Month", 0, 20, 2)
            submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = {
                "age": age,
                "balance": balance,
                "num_accounts": num_accounts,
                "last_login_days": last_login_days,
                "satisfaction_score": satisfaction_score,
                "profession": profession,
                "phase": phase,
                "gender": gender,
                "region": region,
                "risk_profile": risk_profile,
                "logins_per_month": logins_per_month,
            }
            model = st.session_state["model"]
            if model.is_trained:
                segment = model.predict_segment(input_data)
                st.write(f"**Predicted Segment:** {segment}")
            else:
                st.warning(
                    "Please train the segmentation model first on the 'Cluster Training' tab."
                )

    def run(self):
        st.set_page_config(
            page_title="Superannuation Member Segmentation", layout="wide"
        )
        st.header("Superannuation Member Segmentation Explorer")
        st.metric(label="Number of members:", value=f"{self.config.data.n_member:,}")
        self.setup_session_state()

        with st.sidebar:
            self.sidebar_predict_form()

        tab1, tab2, tab3, tab4, tab5, tab6, tab_readme = st.tabs(
            [
                "🧠 Cluster Training",
                "🔍 Data Sample",
                "📉 Pairwise plots",
                "📊 Segment Distributions",
                "📈 Cluster Visualisation",
                "🧪 Cohort Sensitivity",
                "📖 README",
            ]
        )

        with tab1:
            st.info(
                f"Model last trained: {st.session_state['model_metadata']['train_time']}"
            )
            stats = st.session_state["model"].train(st.session_state["data"])
            st.session_state["fit_stats"] = stats
            st.toast("Segmentation model trained!")
            if st.session_state["fit_stats"]:
                st.metric(
                    label="Silhouette Score",
                    value=f"{float(st.session_state['fit_stats']['silhouette']):.2f}",
                    help="Silhouette Score measures how well members fit within their segment (ranges from -1 to 1; higher is better for clustering quality).",
                )
                st.subheader("Cluster Sizes")
                st.dataframe(st.session_state["fit_stats"]["cluster_sizes"])
                display_markdown_file("docs/metrics.md")

        with tab2:
            n_sample = self.config.data.get("n_sample", 20)
            st.subheader(f"Sample of Synthetic Member Data: {n_sample} rows")
            st.dataframe(
                st.session_state["data"].sample(n_sample),
                hide_index=True,
            )

        with tab3:
            pairwise_features_default = [
                "age",
                "balance",
                "num_accounts",
                "last_login_days",
                "satisfaction_score",
            ]
            pairwise_features = st.multiselect(
                "Select features for pairwise interaction plot:",
                options=pairwise_features_default,
                default=pairwise_features_default,
            )
            backend = self.config.data.get("pairplot_backend", "seaborn")
            max_sample = self.config.data.get("max_pairplot_sample", 1000)
            df = st.session_state["data"].sample(
                min(len(st.session_state["data"]), max_sample)
            )
            if backend == "seaborn":
                fig = sns.pairplot(
                    df[pairwise_features + ["segment"]], hue="segment", diag_kind="hist"
                )
                st.pyplot(fig)
            else:
                fig = px.scatter_matrix(
                    df,
                    dimensions=pairwise_features,
                    color="segment",
                    symbol="segment",
                    title="Scatter Matrix of Member Data",
                    labels={
                        col: col.replace("_", " ").title()
                        for col in pairwise_features + ["segment"]
                    },
                    height=800,
                )
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header("Feature Distributions by Segment")
            df = st.session_state["data"]
            numeric_features = [
                "age",
                "balance",
                "num_accounts",
                "last_login_days",
                "satisfaction_score",
            ]
            for feature in numeric_features:
                st.subheader(
                    f"Distribution of {feature.replace('_', ' ').capitalize()}"
                )
                fig = px.histogram(
                    df,
                    x=feature,
                    color="segment",
                    barmode="group",
                    nbins=30,
                    title=f"{feature.replace('_', ' ').capitalize()} Distribution by Segment",
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.header("Cluster Visualisation (e.g., PCA/t-SNE)")
            model = st.session_state["model"]
            if not model.is_trained:
                st.warning(
                    "Please train the segmentation model first on the 'Cluster Training' tab."
                )
            else:
                fig = model.visualise_clusters(st.session_state["data"])
                st.plotly_chart(fig, use_container_width=True)

        with tab6:
            st.header("Age Cohort Sensitivity Analysis")
            model = st.session_state["model"]
            df = st.session_state["data"]

            default_cohorts = [(18, 30), (30, 40), (40, 50), (50, 65), (65, 120)]
            n_clusters = st.number_input("Clusters per cohort", 2, 8, model.n_clusters)
            shifts = st.multiselect(
                "Boundary Shifts (years)", [-2, -1, 0, 1, 2], default=[-2, -1, 0, 1, 2]
            )

            if st.button("Run Sensitivity Analysis"):
                results_df = model.cohort_sensitivity_analysis(
                    df,
                    base_cohorts=default_cohorts,
                    boundary_shifts=shifts,
                    n_clusters=n_clusters,
                )
                st.session_state["sensitivity_results"] = results_df

            if "sensitivity_results" in st.session_state:
                results_df = st.session_state["sensitivity_results"]
                st.dataframe(results_df)
                st.subheader("Silhouette Scores by Shift and Cohort")
                pivot = results_df.pivot(
                    index="shift", columns="cohort", values="silhouette"
                )
                st.line_chart(pivot)

        with tab_readme:
            display_markdown_file(
                "README.md", remove_title=getattr(self.config.readme, "title", None)
            )


# --- Run the App ---

if __name__ == "__main__":
    app = SuperSegmentApp()
    app.run()
