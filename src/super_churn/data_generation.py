import duckdb
import numpy as np
import pandas as pd
from faker import Faker


def generate_age(config) -> int:
    return int(
        np.clip(
            np.random.normal(loc=config["age"]["mean"], scale=config["age"]["std"]),
            config["age"]["min"],
            config["age"]["max"],
        )
    )


def generate_balance(config) -> int:
    balance = int(
        np.random.lognormal(
            mean=config["balance"]["mean"], sigma=config["balance"]["sigma"]
        )
    )
    return np.clip(balance, config["balance"]["min"], config["balance"]["max"])


def generate_num_accounts(config) -> int:
    return np.random.choice(
        config["num_accounts"]["choices"], p=config["num_accounts"]["probabilities"]
    )


def generate_last_login_days(config) -> int:
    return int(
        np.clip(
            np.random.exponential(scale=config["last_login_days"]["scale"]),
            config["last_login_days"]["min"],
            config["last_login_days"]["max"],
        )
    )


def generate_satisfaction_score(config) -> int:
    return np.random.choice(
        config["satisfaction_score"]["choices"],
        p=config["satisfaction_score"]["probabilities"],
    )


def compute_churn_probability(
    num_accounts, last_login_days, satisfaction_score, age, balance, config
) -> float:
    churn_cfg = config["churn_model"]
    logit = (
        churn_cfg["beta_num_accounts"]
        * (num_accounts - churn_cfg["center_num_accounts"])
        + churn_cfg["beta_last_login_days"]
        * (last_login_days - churn_cfg["center_last_login_days"])
        + churn_cfg["beta_satisfaction_score"]
        * (satisfaction_score - churn_cfg["center_satisfaction_score"])
        + churn_cfg["beta_age"] * (age - churn_cfg["center_age"])
        + churn_cfg["beta_balance"] * (balance - churn_cfg["center_balance"])
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn_prob = churn_prob * churn_cfg["scaling_factor"]
    return churn_prob


def generate_churn(
    num_accounts, last_login_days, satisfaction_score, age, balance, config
) -> int:
    churn_prob = compute_churn_probability(
        num_accounts, last_login_days, satisfaction_score, age, balance, config
    )
    return int(np.random.rand() < churn_prob)


class MemberDataGenerator:
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

    def generate(_self, n_member=None) -> pd.DataFrame:
        n = n_member if n_member is not None else _self.config["data"]["n_member"]
        data = []
        for _ in range(n):
            age = generate_age(_self.config)
            balance = generate_balance(_self.config)
            num_accounts = generate_num_accounts(_self.config)
            last_login_days = generate_last_login_days(_self.config)
            satisfaction_score = generate_satisfaction_score(_self.config)
            name = _self.fake.name()
            email = _self.make_au_email(name)
            churned = generate_churn(
                num_accounts,
                last_login_days,
                satisfaction_score,
                age,
                balance,
                _self.config,
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

    def get_or_generate_member_data(n_member, generator, config):
        db_path = config["data"]["member_data_db_path"]
        table = config["data"]["member_data_table"]
        conn = duckdb.connect(db_path)
        # Check if table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if (table,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count >= n_member:
                df = conn.execute(
                    f"SELECT * FROM {table} USING SAMPLE {n_member} ROWS"
                ).df()
                conn.close()
                return df
        # Otherwise, generate new data and store
        df = generator.generate(n_member=n_member)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        conn.close()
        return df
