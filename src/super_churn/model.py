from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from loguru import logger


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
            "report": classification_report(
                self.y_test, self.y_pred, output_dict=True, zero_division=0
            ),
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
