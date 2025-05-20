from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


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


def generate_profession(config) -> str:
    return np.random.choice(
        config["profession"]["choices"],
        p=config["profession"]["probabilities"],
    )


def generate_phase(config) -> str:
    return np.random.choice(
        config["phase"]["choices"],
        p=config["phase"]["probabilities"],
    )


def generate_gender(config) -> str:
    return np.random.choice(
        config["gender"]["choices"],
        p=config["gender"]["probabilities"],
    )


def generate_region(config) -> str:
    return np.random.choice(
        config["region"]["choices"],
        p=config["region"]["probabilities"],
    )


def generate_risk_profile(config) -> str:
    return np.random.choice(
        config["risk_profile"]["choices"],
        p=config["risk_profile"]["probabilities"],
    )


def generate_contrib_freq(config) -> str:
    return np.random.choice(
        config["contrib_freq"]["choices"],
        p=config["contrib_freq"]["probabilities"],
    )


def generate_logins_per_month(config) -> int:
    return int(
        np.clip(
            np.random.poisson(lam=config["logins_per_month"]["mean"]),
            config["logins_per_month"]["min"],
            config["logins_per_month"]["max"],
        )
    )


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

    def generate(self, n_member=None) -> pd.DataFrame:
        n = n_member if n_member is not None else self.config["data"]["n_member"]
        data = []
        for _ in range(n):
            age = generate_age(self.config)
            balance = generate_balance(self.config)
            num_accounts = generate_num_accounts(self.config)
            last_login_days = generate_last_login_days(self.config)
            satisfaction_score = generate_satisfaction_score(self.config)
            profession = generate_profession(self.config)
            phase = generate_phase(self.config)
            gender = generate_gender(self.config)
            region = generate_region(self.config)
            risk_profile = generate_risk_profile(self.config)
            contrib_freq = generate_contrib_freq(self.config)
            logins_per_month = generate_logins_per_month(self.config)
            name = self.fake.name()
            email = self.make_au_email(name)
            data.append(
                {
                    "name": name,
                    "email": email,
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
                    "contrib_freq": contrib_freq,
                    "logins_per_month": logins_per_month,
                }
            )
        return pd.DataFrame(data)


def get_or_generate_member_data(n_member, generator, config):
    db_path = Path(config["data"]["member_data_db_path"])
    table = config["data"]["member_data_table"]

    # Ensure the parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using DuckDB database at {db_path.resolve()}.")

    conn = duckdb.connect(str(db_path))
    try:
        # Check if table exists
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"Table '{table}' exists with {count} rows.")
            if count >= n_member:
                logger.info(
                    f"Loading {n_member} members from cached DuckDB table '{table}'."
                )
                df = conn.execute(f"SELECT * FROM {table} LIMIT {n_member}").df()
                return df
            else:
                logger.warning(
                    f"Table '{table}' has insufficient rows ({count} < {n_member}); regenerating data."
                )

        # Otherwise, generate new data and store
        logger.info(
            f"Generating new member data and storing in DuckDB table '{table}'."
        )
        df = generator.generate(n_member=n_member)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        logger.success(f"Stored {n_member} members in DuckDB table '{table}'.")
        return df
    finally:
        conn.close()
        logger.debug("DuckDB connection closed.")
