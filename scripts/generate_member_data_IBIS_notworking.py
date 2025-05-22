import os
from pathlib import Path

import duckdb
import hydra
import logfire
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from faker import Faker
from omegaconf import DictConfig, OmegaConf

load_dotenv()

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))


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

        # Cluster setup
        self.cluster_names = config["clusters"]["segment_names"]
        self.cluster_probs = config["clusters"]["segment_probs"]
        if not np.isclose(sum(self.cluster_probs), 1.0):
            raise ValueError("Cluster probabilities must sum to 1")

    def _get_cluster_config(self, cluster_name: str) -> dict:
        """Get feature parameters for a specific cluster"""
        return self.config["clusters"][cluster_name]

    def make_au_email(self, name: str) -> str:
        domains = self.config["email"]["domains"]
        username = name.lower().replace(" ", ".").replace("'", "").replace("-", "")
        domain = np.random.choice(domains)
        return f"{username}@{domain}"

    def generate(self, n_member=None) -> pd.DataFrame:
        n = n_member if n_member is not None else self.config["data"]["n_member"]
        data = []
        for _ in range(n):
            # Assign to a cluster first
            cluster_name = np.random.choice(self.cluster_names, p=self.cluster_probs)
            cluster_config = self._get_cluster_config(cluster_name)

            # Generate features using cluster-specific parameters
            age = generate_age(cluster_config)
            balance = generate_balance(cluster_config)
            num_accounts = generate_num_accounts(cluster_config)
            last_login_days = generate_last_login_days(
                self.config
            )  # Use global for non-clustered features
            satisfaction_score = generate_satisfaction_score(self.config)
            profession = generate_profession(cluster_config)
            phase = generate_phase(self.config)
            gender = generate_gender(self.config)
            region = generate_region(self.config)
            risk_profile = generate_risk_profile(cluster_config)
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
                    "true_segment": cluster_name,  # Ground truth for validation
                }
            )

        return pd.DataFrame(data)


def get_or_generate_member_data(n_member, generator, config):
    db_path = Path(config["data"]["member_data_db_path"])
    table = config["data"]["member_data_table"]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    logfire.info(f"Using DuckDB database at {db_path.resolve()}.")

    conn = duckdb.connect(str(db_path))
    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logfire.info(f"Table '{table}' exists with {count} rows.")
            if count >= n_member:
                logfire.info(
                    f"Loading {n_member} members from cached DuckDB table '{table}'."
                )
                df = conn.execute(f"SELECT * FROM {table} LIMIT {n_member}").df()
                return df
            else:
                logfire.warning(
                    f"Table '{table}' has insufficient rows ({count} < {n_member}); regenerating data."
                )

        logfire.info(
            f"Generating new member data and storing in DuckDB table '{table}'."
        )
        df = generator.generate(n_member=n_member)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        logfire.info(f"Stored {n_member} members in DuckDB table '{table}'.")
        return df
    finally:
        conn.close()
        logfire.debug("DuckDB connection closed.")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Convert OmegaConf to plain dict for numpy/faker compatibility
    config = OmegaConf.to_container(cfg, resolve=True)
    generator = MemberDataGenerator(config)
    n_member = config["data"]["n_member"]
    df = get_or_generate_member_data(n_member, generator, config)
    output_file = config["data"].get("output_file", "data/members.parquet")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file)
    logfire.info(f"Saved members.parquet", path=output_file)
    print(f"Generated {len(df)} members and saved to {output_file}")


if __name__ == "__main__":
    main()
