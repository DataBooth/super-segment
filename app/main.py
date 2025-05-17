import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from faker import Faker
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- Utility Functions ---


def generate_age() -> int:
    """Generate age using a normal distribution, clipped to 18-65."""
    return int(np.clip(np.random.normal(loc=40, scale=10), 18, 65))


def generate_balance() -> int:
    """Generate balance using a log-normal distribution, clipped to 1,000–1,000,000."""
    balance = int(np.random.lognormal(mean=10, sigma=0.7))
    return np.clip(balance, 1000, 1_000_000)


def generate_num_accounts() -> int:
    """Generate number of accounts using ATO-based categorical probabilities."""
    return np.random.choice([1, 2, 3, 4], p=[0.63, 0.25, 0.09, 0.03])


def generate_last_login_days() -> int:
    """Generate days since last login using an exponential distribution, capped at 365."""
    return int(np.clip(np.random.exponential(scale=90), 0, 365))


def generate_satisfaction_score() -> int:
    """Generate satisfaction score with a realistic, right-skewed distribution."""
    return np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30])


def compute_churn_probability(
    num_accounts, last_login_days, satisfaction_score, age, balance
) -> float:
    """Compute churn probability using a logistic-like model, scaled to produce realistic churn rates."""
    logit = (
        0.8 * (num_accounts - 1)
        + 0.01 * (last_login_days - 90)
        + -0.7 * (satisfaction_score - 3)
        + 0.015 * (age - 40)
        + 0.000005 * (balance - 50000)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn_prob = churn_prob * 0.28  # scale to target ~19% churn rate
    return churn_prob


def generate_churn(
    num_accounts, last_login_days, satisfaction_score, age, balance
) -> int:
    """Generate churn label as a Bernoulli sample from the computed probability."""
    churn_prob = compute_churn_probability(
        num_accounts, last_login_days, satisfaction_score, age, balance
    )
    return int(np.random.rand() < churn_prob)


def display_markdown_file(
    filepath: str, encoding: str = "utf-8", warn_if_missing: bool = True
):
    """Display a markdown file in Streamlit, or show a warning if the file does not exist."""
    path = Path(filepath)
    if path.exists():
        st.markdown(path.read_text(encoding=encoding))
    elif warn_if_missing:
        st.warning(f"{path.name} file not found in the project directory.")


# --- Data and Model Classes ---


class SuperannuationDataGenerator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.fake = Faker()
        Faker.seed(config["data"]["random_seed"])
        np.random.seed(config["data"]["random_seed"])

    def make_au_email(self, name: str) -> str:
        domains = self.config["email"]["domains"]
        username = name.lower().replace(" ", ".").replace("'", "").replace("-", "")
        domain = np.random.choice(domains)
        return f"{username}@{domain}"

    def generate(self) -> pd.DataFrame:
        data = []
        for _ in range(self.config["data"]["n_samples"]):
            age = generate_age()
            balance = generate_balance()
            num_accounts = generate_num_accounts()
            last_login_days = generate_last_login_days()
            satisfaction_score = generate_satisfaction_score()
            name = self.fake.name()
            email = self.make_au_email(name)
            churned = generate_churn(
                num_accounts, last_login_days, satisfaction_score, age, balance
            )
            data.append(
                {
                    "name": name,
                    "email": email,
                    "age": age,
                    "balance": balance,
                    "num_accounts": num_accounts,
                    "last_login_days": last_login_days,
                    "satisfaction_score": satisfaction_score,
                    "churned": churned,
                }
            )
        return pd.DataFrame(data)


class SuperannuationChurnModel:
    """Trains and predicts churn using logistic regression on superannuation data."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)
        self.features = [
            "age",
            "balance",
            "num_accounts",
            "last_login_days",
            "satisfaction_score",
        ]
        self.is_trained = False
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def train(self, df: pd.DataFrame) -> dict:
        X = df[self.features]
        y = df["churned"]
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.is_trained = True
        logger.success("Model trained.")
        return {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "report": classification_report(self.y_test, self.y_pred, output_dict=True),
        }

    def predict_proba(self, input_data: dict) -> float:
        X_input = pd.DataFrame([input_data])[self.features]
        prob = self.model.predict_proba(X_input)[0][1]
        logger.info(f"Predicted churn probability: {prob:.2%}")
        return prob

    def get_residuals_df(self) -> pd.DataFrame:
        if (
            not self.is_trained
            or self.X_test is None
            or self.y_test is None
            or self.y_pred is None
        ):
            raise ValueError("Model must be trained before getting residuals.")
        residuals = self.y_test - self.y_pred
        return pd.DataFrame(
            {"Predicted": self.y_pred, "Actual": self.y_test, "Residual": residuals}
        )


# --- Main App Class ---


class SuperChurnApp:
    def __init__(self):
        self.config = self.load_config()
        self.data_generator = SuperannuationDataGenerator(self.config)
        self.model = None

    @staticmethod
    def load_config():
        with open("config.toml", "rb") as f:
            return tomllib.load(f)

    @st.cache_data(show_spinner="Generating synthetic data...")
    def get_synthetic_data(_self, config: dict) -> pd.DataFrame:
        generator = SuperannuationDataGenerator(config)
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
        st.title("Superannuation Churn Predictor")
        self.setup_session_state()

        with st.sidebar:
            self.sidebar_predict_form()

        tab1, tab2, tab4, tab5, tab_readme = st.tabs(
            [
                "🧠 Model Training",
                "🔍 Data Sample",
                "📊 Data Distributions",
                "📈 Model Fit Visualisation",
                "📖 Readme",
            ]
        )

        with tab1:
            st.header("Fit the Model and View Statistics")
            if st.button("Fit Model"):
                stats = st.session_state["model"].train(st.session_state["data"])
                st.session_state["fit_stats"] = stats
                st.toast("Model trained!")
            if st.session_state["fit_stats"]:
                st.subheader("Model Accuracy")
                st.write(f"{st.session_state['fit_stats']['accuracy']:.2%}")
                st.subheader("Classification Report")
                report_dict = st.session_state["fit_stats"]["report"]
                report_df = pd.DataFrame(report_dict).transpose()
                report_df = report_df.apply(pd.to_numeric, errors="coerce")
                st.dataframe(report_df.style.format("{:.2f}"))
                display_markdown_file("metrics.md")

        with tab2:
            st.header("Sample of Synthetic Data")
            st.dataframe(st.session_state["data"].sample(10))

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
            display_markdown_file("maths.md")
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
            display_markdown_file("README.md")


# --- Run the App ---

if __name__ == "__main__":
    app = SuperChurnApp()
    app.run()
