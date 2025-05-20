import pytest
import pandas as pd
from super_segment.data_generation import MemberDataGenerator


@pytest.fixture
def config():
    # Minimal config for testing
    return {
        "data": {"n_member": 10, "random_seed": 42},
        "age": {"mean": 40, "std": 10, "min": 25, "max": 65},
        "balance": {"mean": 10, "sigma": 0.7, "min": 20000, "max": 300000},
        "num_accounts": {"choices": [1, 2], "probabilities": [0.7, 0.3]},
        "last_login_days": {"scale": 90, "min": 1, "max": 180},
        "satisfaction_score": {"choices": [1, 2, 3], "probabilities": [0.2, 0.5, 0.3]},
        "profession": {"choices": ["A", "B"], "probabilities": [0.5, 0.5]},
        "phase": {
            "choices": ["Accumulation", "Retirement"],
            "probabilities": [0.7, 0.3],
        },
        "gender": {"choices": ["Male", "Female"], "probabilities": [0.5, 0.5]},
        "region": {"choices": ["NSW", "VIC"], "probabilities": [0.5, 0.5]},
        "risk_profile": {
            "choices": ["Conservative", "Aggressive"],
            "probabilities": [0.5, 0.5],
        },
        "contrib_freq": {"choices": ["Monthly", "Yearly"], "probabilities": [0.8, 0.2]},
        "logins_per_month": {"mean": 2, "min": 0, "max": 10},
        "email": {"domains": ["test.com"]},
    }


def test_member_data_generator(config):
    gen = MemberDataGenerator(config)
    df = gen.generate(n_member=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    # Check columns
    expected_cols = {
        "name",
        "email",
        "age",
        "balance",
        "num_accounts",
        "last_login_days",
        "satisfaction_score",
        "profession",
        "phase",
        "gender",
        "region",
        "risk_profile",
        "contrib_freq",
        "logins_per_month",
    }
    assert expected_cols.issubset(df.columns)
