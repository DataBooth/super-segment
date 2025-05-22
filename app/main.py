import streamlit as st
import plotly.express as px
import seaborn as sns
import duckdb
from super_segment.model import get_or_train_model
from super_segment.utils import display_markdown_file
from super_segment.app_config import AppConfig
import datetime

def format_train_time(train_time_str):
    # Parse the timestamp (assuming ISO format, e.g., '2025-05-27T08:35:00')
    train_time = datetime.datetime.fromisoformat(train_time_str)
    now = datetime.datetime.now(train_time.tzinfo) if train_time.tzinfo else datetime.datetime.now()
    delta = now - train_time

    # Format date as "Mon 27 May 2025 8:35AM"
    formatted_date = train_time.strftime("%a %d %b %Y %-I:%M%p")

    # Format "xx hours ago" or "yy days ago"
    if delta.days >= 2:
        ago = f"{delta.days} days ago"
    elif delta.days == 1:
        ago = "yesterday"
    elif delta.seconds >= 7200:
        ago = f"{delta.seconds // 3600} hours ago"
    elif delta.seconds >= 3600:
        ago = "an hour ago"
    elif delta.seconds >= 120:
        ago = f"{delta.seconds // 60} minutes ago"
    else:
        ago = "just now"

    return f"**Model last trained: {formatted_date} ({ago})**"


class SuperSegmentApp:
    def __init__(self):
        self.config = AppConfig()
        # For convenience, get UI config (sidebar section) once
        self.ui = self.config.get("sidebar", sub_name="ui")

    def get_duckdb_data(self):
        parquet_path = self.config.get("data", "output_file", sub_name="generate")
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{parquet_path}'").df()
        con.close()
        return df

    def setup_session_state(self):
        if "data_full" not in st.session_state:
            st.session_state["data_full"] = self.get_duckdb_data()
        if (
            "model" not in st.session_state
            or "model_metadata" not in st.session_state
            or "fit_stats" not in st.session_state
        ):
            model, metadata, fit_stats = get_or_train_model(
                st.session_state["data_full"]
            )
            st.session_state["model"] = model
            st.session_state["model_metadata"] = metadata
            st.session_state["fit_stats"] = fit_stats
        df = st.session_state["data_full"]
        n_member_train = st.session_state["model_metadata"]["n_member"]
        if "unique_id" in df.columns:
            df = df.sort_values("unique_id")
        st.session_state["data"] = df.iloc[:n_member_train].copy()

    def sidebar_predict_form(self):
        ui = self.ui
        st.header("Predict Segment for a Member")
        with st.form("predict_form"):
            age = st.slider(
                "Age", ui["age"]["min"], ui["age"]["max"], ui["age"]["default"]
            )
            col1, col2 = st.columns(2)
            with col1:
                balance = st.number_input(
                    "Balance",
                    ui["balance"]["min"],
                    ui["balance"]["max"],
                    ui["balance"]["default"],
                    step=ui["balance"]["step"],
                )
            with col2:
                num_accounts = st.selectbox(
                    "Number of Accounts",
                    ui["num_accounts"]["options"],
                    index=list(ui["num_accounts"]["options"]).index(
                        ui["num_accounts"]["default"]
                    ),
                )
            last_login_days = st.slider(
                "Days Since Last Login",
                ui["last_login_days"]["min"],
                ui["last_login_days"]["max"],
                ui["last_login_days"]["default"],
            )
            satisfaction_score = st.selectbox(
                "Satisfaction Score",
                ui["satisfaction_score"]["options"],
                index=list(ui["satisfaction_score"]["options"]).index(
                    ui["satisfaction_score"]["default"]
                ),
            )
            profession = st.selectbox(
                "Profession",
                ui["profession"]["options"],
                index=list(ui["profession"]["options"]).index(
                    ui["profession"]["default"]
                ),
            )
            col3, col4 = st.columns(2)
            with col3:
                gender = st.selectbox(
                    "Gender",
                    ui["gender"]["options"],
                    index=list(ui["gender"]["options"]).index(ui["gender"]["default"]),
                )
            with col4:
                region = st.selectbox(
                    "Region",
                    ui["region"]["options"],
                    index=list(ui["region"]["options"]).index(ui["region"]["default"]),
                )
            col5, col6 = st.columns(2)
            with col5:
                phase = st.selectbox(
                    "Phase",
                    ui["phase"]["options"],
                    index=list(ui["phase"]["options"]).index(ui["phase"]["default"]),
                )
            with col6:
                risk_profile = st.selectbox(
                    "Risk Profile",
                    ui["risk_profile"]["options"],
                    index=list(ui["risk_profile"]["options"]).index(
                        ui["risk_profile"]["default"]
                    ),
                )
            logins_per_month = st.slider(
                "Logins per Month",
                ui["logins_per_month"]["min"],
                ui["logins_per_month"]["max"],
                ui["logins_per_month"]["default"],
            )
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
            page_title=self.config.get(
                "app", "title", default="Superannuation Member Segmentation"
            ),
            layout=self.config.get("app", "layout", default="wide"),
        )
        st.header(
            self.config.get(
                "app", "title", default="Superannuation Member Segmentation"
            )
        )
        self.setup_session_state()
        model_metadata = st.session_state["model_metadata"]
        st.metric(
            label="Number of members (model):", value=f"{model_metadata['n_member']:,}"
        )
        model_train_info = st.empty()
        model_train_info.markdown(
            format_train_time(st.session_state['model_metadata']['train_time'])
        )
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
            col1, col2, _ = st.columns([1, 1, 1], gap="large")
            with col2:
                n_clusters = st.slider(
                    "Number of groups (clusters)", min_value=2, max_value=6, value=4
                )
            with col1:
                if st.button("Train Model"):
                    model, metadata, fit_stats = get_or_train_model(
                        st.session_state["data_full"], n_clusters=n_clusters
                    )
                    st.session_state["model"] = model
                    st.session_state["model_metadata"] = metadata
                    st.session_state["fit_stats"] = fit_stats
                    df = st.session_state["data_full"]
                    n_member_train = metadata["n_member"]
                    if "unique_id" in df.columns:
                        df = df.sort_values("unique_id")
                    st.session_state["data"] = df.iloc[:n_member_train].copy()
                    model_train_info.markdown(
                        format_train_time(st.session_state['model_metadata']['train_time'])
                    )
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
            n_sample = self.config.get(
                "data", "n_sample", sub_name="generate", default=20
            )
            st.subheader(f"Sample of Synthetic Member Data: {n_sample} rows")
            st.dataframe(st.session_state["data"].sample(n_sample), hide_index=True)

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
            backend = "seaborn"
            max_sample = 1000
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
            readme_file = self.config.get("app", "readme_file", default="README.md")
            display_markdown_file(readme_file)


if __name__ == "__main__":
    app = SuperSegmentApp()
    app.run()
