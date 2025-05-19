import matplotlib.pyplot as plt
from millify import millify
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from super_churn.config import load_config
from super_churn.data_generation import MemberDataGenerator
from super_churn.datastore import get_or_generate_member_data
from super_churn.model import SuperannuationChurnModel
from super_churn.utils import display_markdown_file

# --- App Class ---


class SuperChurnApp:
    def __init__(self):
        self.config = load_config()
        self.data_generator = MemberDataGenerator(self.config)
        self.model = None

    @st.cache_data(show_spinner="Generating synthetic data...")
    def get_synthetic_data(_self, config: dict) -> pd.DataFrame:
        generator = MemberDataGenerator(config)
        return generator.generate()

    def setup_session_state(self):
        if "data" not in st.session_state:
            st.session_state["data"] = self.get_synthetic_data(self.config)
        if "model" not in st.session_state:
            st.session_state["model"] = SuperannuationChurnModel()
        if "fit_stats" not in st.session_state:
            st.session_state["fit_stats"] = None

    def sidebar_predict_form(self):
        st.header("Predict Churn for a New Customer")
        with st.form("predict_form"):
            age = st.slider("Age", 25, 65, 40)
            last_login_days = st.slider("Days Since Last Login", 1, 180, 30)
            balance = st.number_input("Balance", 20000, 300000, 50000, step=1000)
            num_accounts = st.selectbox("Number of Accounts", [1, 2, 3, 4], index=0)
            satisfaction_score = st.selectbox(
                "Satisfaction Score", [1, 2, 3, 4, 5], index=2
            )
            submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = {
                "age": age,
                "balance": balance,
                "num_accounts": num_accounts,
                "last_login_days": last_login_days,
                "satisfaction_score": satisfaction_score,
            }
            model = st.session_state["model"]
            if model.is_trained:
                prob = model.predict_proba(input_data)
                st.write(f"**Predicted churn probability:** {prob:.2%}")
                st.write("**Prediction:**", "Churn" if prob > 0.5 else "No Churn")
            else:
                st.warning("Please train the model first on the 'Model Training' tab.")

    def run(self):
        st.set_page_config(page_title="Superannuation Churn Predictor", layout="wide")
        st.header("Superannuation Member Churn Predictor")
        st.metric(
            label="Number of members:", value=millify(self.config["data"]["n_member"])
        )
        self.setup_session_state()

        with st.sidebar:
            self.sidebar_predict_form()

        tab1, tab2, tab3, tab4, tab5, tab_readme = st.tabs(
            [
                "🧠 Model Training",
                "🔍 Data Sample",
                "📉 Pairwise plots",
                "📊 Data Distributions",
                "📈 Model Fit Visualisation",
                "📖 README",
            ]
        )

        with tab1:
            # st.header("Fit the Model and View Metrics")
            stats = st.session_state["model"].train(st.session_state["data"])
            st.session_state["fit_stats"] = stats
            st.toast("Model trained!")
            if st.session_state["fit_stats"]:
                st.metric(
                    "Model Accuracy", float(st.session_state["fit_stats"]["accuracy"])
                )

                st.subheader("Classification Report")
                report_dict = st.session_state["fit_stats"]["report"]
                report_df = pd.DataFrame(report_dict).transpose()
                report_df = report_df.apply(pd.to_numeric, errors="coerce")
                st.dataframe(report_df.style.format("{:.2f}"))
                display_markdown_file("docs/metrics.md")

        with tab2:
            n_sample = self.config["data"]["n_sample"]
            st.subheader(f"Sample of Synthetic Member Data: {n_sample} rows")
            st.dataframe(
                st.session_state["data"].sample(n_sample),
                hide_index=True,
            )

        with tab3:
            # Select the numerical features you want to include
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
                help="Choose which features to show in the scatterplot matrix.",
            )

            backend = self.config["data"].get("pairplot_backend", "seaborn")
            max_sample = self.config["data"].get("max_pairplot_sample", 1000)
            df = st.session_state["data"].sample(
                min(len(st.session_state["data"]), max_sample)
            )

            # st.header("🔗 Pairwise Plots")

            if backend == "seaborn":
                fig = sns.pairplot(
                    df[pairwise_features + ["churned"]], hue="churned", diag_kind="hist"
                )
                st.pyplot(fig)
            else:
                fig = px.scatter_matrix(
                    df,
                    dimensions=pairwise_features,
                    color="churned",
                    symbol="churned",
                    title="Scatter Matrix of Member Data",
                    labels={
                        col: col.replace("_", " ").title()
                        for col in pairwise_features + ["churned"]
                    },
                    height=800,
                )
                fig.update_traces(
                    diagonal_visible=False
                )  # Optionally hide the diagonal
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header("Feature Distributions")
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
                if feature in ["num_accounts", "satisfaction_score"]:
                    fig = px.histogram(
                        df,
                        x=feature,
                        title=f"{feature.replace('_', ' ').capitalize()} Distribution",
                    )
                    fig.update_xaxes(
                        type="category",
                        categoryorder="total descending",
                    )
                else:
                    fig = px.histogram(
                        df,
                        x=feature,
                        nbins=30,
                        title=f"{feature.replace('_', ' ').capitalize()} Distribution",
                    )
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.header("Model Fit Visualisation")
            display_markdown_file("docs/maths.md")
            model = st.session_state["model"]
            if not model.is_trained:
                st.warning("Please train the model first on the 'Model Training' tab.")
            else:
                residuals_df = model.get_residuals_df()
                # Residuals vs Predicted plot
                fig_residuals = px.scatter(
                    residuals_df,
                    x="Predicted",
                    y="Residual",
                    title="Residuals vs Predicted Values",
                    labels={
                        "Predicted": "Predicted Churn (0 or 1)",
                        "Residual": "Residual (Actual - Predicted)",
                    },
                    trendline="ols",
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
                # Actual vs Predicted plot
                fig_actual_pred = px.scatter(
                    residuals_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Churn",
                    labels={
                        "Actual": "Actual Churn (0 or 1)",
                        "Predicted": "Predicted Churn (0 or 1)",
                    },
                )
                fig_actual_pred.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(color="red", dash="dash"),
                )
                st.plotly_chart(fig_actual_pred, use_container_width=True)

        with tab_readme:
            display_markdown_file(
                "README.md", remove_title=self.config["readme"]["title"]
            )


# --- Run the App ---

if __name__ == "__main__":
    app = SuperChurnApp()
    app.run()
