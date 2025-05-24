from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger
from super_segment.project_config import ProjectConfig  

def generate_age(config: ProjectConfig) -> int:
    return int(
        np.clip(
            np.random.normal(
                loc=config.get("age", "mean", file="generate"),
                scale=config.get("age", "std", file="generate")
            ),
            config.get("age", "min", file="generate"),
            config.get("age", "max", file="generate"),
        )
    )

def generate_balance(config: ProjectConfig) -> int:
    balance = int(
        np.random.lognormal(
            mean=config.get("balance", "mean", file="generate"),
            sigma=config.get("balance", "sigma", file="generate")
        )
    )
    return np.clip(
        balance,
        config.get("balance", "min", file="generate"),
        config.get("balance", "max", file="generate")
    )

def generate_num_accounts(config: ProjectConfig) -> int:
    return np.random.choice(
        config.get("num_accounts", "choices", file="generate"),
        p=config.get("num_accounts", "probabilities", file="generate")
    )

def generate_last_login_days(config: ProjectConfig) -> int:
    return int(
        np.clip(
            np.random.exponential(scale=config.get("last_login_days", "scale", file="generate")),
            config.get("last_login_days", "min", file="generate"),
            config.get("last_login_days", "max", file="generate"),
        )
    )

def generate_satisfaction_score(config: ProjectConfig) -> int:
    return np.random.choice(
        config.get("satisfaction_score", "choices", file="generate"),
        p=config.get("satisfaction_score", "probabilities", file="generate")
    )

def generate_profession(config: ProjectConfig) -> str:
    return np.random.choice(
        config.get("profession", "choices", file="generate"),
        p=config.get("profession", "probabilities", file="generate")
    )

def generate_phase(config: ProjectConfig) -> str:
    return np.random.choice(
        config.get("phase", "choices", file="generate"),
        p=config.get("phase", "probabilities", file="generate")
    )

def generate_gender(config: ProjectConfig) -> str:
    return np.random.choice(
        config.get("gender", "choices", file="generate"),
        p=config.get("gender", "probabilities", file="generate")
    )

def generate_region(config: ProjectConfig) -> str:
    return np.random.choice(
        config.get("region", "choices", file="generate"),
        p=config.get("region", "probabilities", file="generate")
    )

def generate_risk_profile(config: ProjectConfig) -> str:
    return np.random.choice(
        config.get("risk_profile", "choices", file="generate"),
        p=config.get("risk_profile", "probabilities", file="generate")
    )

def generate_contrib_freq(config: ProjectConfig) -> str:
    return np.random.choice(
        config.get("contrib_freq", "choices", file="generate"),
        p=config.get("contrib_freq", "probabilities", file="generate")
    )

def generate_logins_per_month(config: ProjectConfig) -> int:
    return int(
        np.clip(
            np.random.poisson(lam=config.get("logins_per_month", "mean", file="generate")),
            config.get("logins_per_month", "min", file="generate"),
            config.get("logins_per_month", "max", file="generate"),
        )
    )

class MemberDataGenerator:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.fake = Faker()
        random_seed = self.config.get("data", "random_seed", file="generate")
        Faker.seed(random_seed)
        np.random.seed(random_seed)

        # Cluster setup
        self.cluster_names = self.config.get("clusters", "segment_names", file="generate")
        self.cluster_probs = self.config.get("clusters", "segment_probs", file="generate")
        if not np.isclose(sum(self.cluster_probs), 1.0):
            raise ValueError("Cluster probabilities must sum to 1")

    def _get_cluster_config(self, cluster_name: str) -> dict:
        """Get feature parameters for a specific cluster from the config."""
        return self.config.get("clusters", cluster_name, file="generate")

    def make_au_email(self, name: str) -> str:
        domains = self.config.get("email", "domains", file="generate")
        username = name.lower().replace(" ", ".").replace("'", "").replace("-", "")
        domain = np.random.choice(domains)
        return f"{username}@{domain}"

    def generate(self, n_member=None) -> pd.DataFrame:
        n = n_member if n_member is not None else self.config.get("data", "n_member", file="generate")
        data = []
        for _ in range(n):
            # Assign to a cluster first
            cluster_name = np.random.choice(self.cluster_names, p=self.cluster_probs)
            cluster_config = self._get_cluster_config(cluster_name)

            # Generate features using cluster-specific parameters
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
                    "true_segment": cluster_name,  # Ground truth for validation
                }
            )

        return pd.DataFrame(data)

def get_or_generate_member_data(n_member: int, generator: MemberDataGenerator, config: ProjectConfig) -> pd.DataFrame:
    db_path = Path(config.get("data", "member_data_db_path", file="generate"))
    table = config.get("data", "member_data_table", file="generate")

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
