import pytest
import numpy as np
import pandas as pd

from main import (
    generate_age,
    generate_balance,
    generate_num_accounts,
    generate_last_login_days,
    generate_satisfaction_score,
    compute_churn_probability,
    generate_churn,
    MemberDataGenerator,
    SuperannuationChurnModel,
)


# Example minimal config for testing
@pytest.fixture
def config():
    return {
        "age": {"mean": 40, "std": 10, "min": 25, "max": 65},
        "balance": {"mean": 10, "sigma": 0.7, "min": 20000, "max": 300000},
        "num_accounts": {
            "choices": [1, 2, 3, 4],
            "probabilities": [0.63, 0.25, 0.09, 0.03],
        },
        "last_login_days": {"scale": 90, "min": 1, "max": 180},
        "satisfaction_score": {
            "choices": [1, 2, 3, 4, 5],
            "probabilities": [0.05, 0.10, 0.20, 0.35, 0.30],
        },
        "churn_model": {
            "beta_num_accounts": 0.8,
            "center_num_accounts": 1,
            "beta_last_login_days": 0.01,
            "center_last_login_days": 90,
            "beta_satisfaction_score": -0.7,
            "center_satisfaction_score": 3,
            "beta_age": 0.015,
            "center_age": 40,
            "beta_balance": 0.000005,
            "center_balance": 50000,
            "scaling_factor": 0.28,
        },
        "data": {"random_seed": 42, "n_member": 100},
        "email": {"domains": ["test.com"]},
    }


def test_generate_age(config):
    for _ in range(100):
        age = generate_age(config)
        assert config["age"]["min"] <= age <= config["age"]["max"]


def test_generate_balance(config):
    for _ in range(100):
        bal = generate_balance(config)
        assert config["balance"]["min"] <= bal <= config["balance"]["max"]


def test_generate_num_accounts(config):
    for _ in range(100):
        n = generate_num_accounts(config)
        assert n in config["num_accounts"]["choices"]


def test_generate_last_login_days(config):
    for _ in range(100):
        days = generate_last_login_days(config)
        assert (
            config["last_login_days"]["min"] <= days <= config["last_login_days"]["max"]
        )


def test_generate_satisfaction_score(config):
    for _ in range(100):
        score = generate_satisfaction_score(config)
        assert score in config["satisfaction_score"]["choices"]


def test_compute_churn_probability(config):
    prob = compute_churn_probability(2, 100, 4, 45, 100000, config)
    assert 0.0 <= prob <= 1.0


def test_generate_churn(config):
    churn = generate_churn(2, 100, 4, 45, 100000, config)
    assert churn in (0, 1)


def test_member_data_generator(config):
    gen = MemberDataGenerator(config)
    df = gen.generate(10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert set(
        [
            "name",
            "email",
            "age",
            "balance",
            "num_accounts",
            "last_login_days",
            "satisfaction_score",
            "churned",
        ]
    ).issubset(df.columns)


def test_model_training_and_prediction(config):
    gen = MemberDataGenerator(config)
    df = gen.generate(100)
    model = SuperannuationChurnModel()
    stats = model.train(df)
    assert "accuracy" in stats
    assert 0 <= stats["accuracy"] <= 1
    input_data = df.iloc[0][
        ["age", "balance", "num_accounts", "last_login_days", "satisfaction_score"]
    ].to_dict()
    prob = model.predict_proba(input_data)
    assert 0 <= prob <= 1
