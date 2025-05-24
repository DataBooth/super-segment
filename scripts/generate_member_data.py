from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
from faker import Faker
from loguru import logger

from super_segment.project_config import ProjectConfig  # Updated import


def make_au_email(name, domains):
    username = name.lower().replace(" ", ".").replace("'", "").replace("-", "")
    domain = np.random.choice(domains)
    return f"{username}@{domain}"


def count_parquet_rows(parquet_path: str) -> int:
    if not Path(parquet_path).exists():
        return 0
    con = duckdb.connect()
    try:
        result = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()
        return result[0] if result else 0
    except Exception as e:
        logger.warning(f"Could not count rows in {parquet_path}: {e}")
        return 0
    finally:
        con.close()


class MemberDataGenerator:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.fake = Faker()
        random_seed = self.config.get("data", "random_seed", file="generate")
        Faker.seed(random_seed)
        np.random.seed(random_seed)
        self.cluster_names = self.config.get("clusters", "segment_names", file="generate")
        self.cluster_probs = self.config.get("clusters", "segment_probs", file="generate")
        self.clusters = self.config.get("clusters", file="generate")
        self.email_domains = self.config.get("email", "domains", file="generate")

    def generate_member(self) -> dict:
        cluster_name = np.random.choice(self.cluster_names, p=self.cluster_probs)
        cluster_config = self.clusters[cluster_name]

        age = int(
            np.clip(
                np.random.normal(
                    loc=cluster_config["age"]["mean"],
                    scale=cluster_config["age"]["std"],
                ),
                cluster_config["age"]["min"],
                cluster_config["age"]["max"],
            )
        )

        balance = int(
            np.random.lognormal(
                mean=cluster_config["balance"]["mean"],
                sigma=cluster_config["balance"]["sigma"],
            )
        )
        balance = np.clip(
            balance, cluster_config["balance"]["min"], cluster_config["balance"]["max"]
        )

        num_accounts = np.random.choice(
            cluster_config["num_accounts"]["choices"],
            p=cluster_config["num_accounts"]["probabilities"],
        )

        last_login_days = int(
            np.clip(
                np.random.exponential(scale=self.config.get("last_login_days", "scale", file="generate")),
                self.config.get("last_login_days", "min", file="generate"),
                self.config.get("last_login_days", "max", file="generate"),
            )
        )

        satisfaction_score = np.random.choice(
            self.config.get("satisfaction_score", "choices", file="generate"),
            p=self.config.get("satisfaction_score", "probabilities", file="generate"),
        )

        profession = np.random.choice(
            cluster_config["profession"]["choices"],
            p=cluster_config["profession"]["probabilities"],
        )

        phase = np.random.choice(
            self.config.get("phase", "choices", file="generate"),
            p=self.config.get("phase", "probabilities", file="generate"),
        )

        gender = np.random.choice(
            self.config.get("gender", "choices", file="generate"),
            p=self.config.get("gender", "probabilities", file="generate"),
        )

        region = np.random.choice(
            self.config.get("region", "choices", file="generate"),
            p=self.config.get("region", "probabilities", file="generate"),
        )

        risk_profile = np.random.choice(
            cluster_config["risk_profile"]["choices"],
            p=cluster_config["risk_profile"]["probabilities"],
        )

        contrib_freq = np.random.choice(
            self.config.get("contrib_freq", "choices", file="generate"),
            p=self.config.get("contrib_freq", "probabilities", file="generate"),
        )

        logins_per_month = int(
            np.clip(
                np.random.poisson(lam=self.config.get("logins_per_month", "mean", file="generate")),
                self.config.get("logins_per_month", "min", file="generate"),
                self.config.get("logins_per_month", "max", file="generate"),
            )
        )

        name = self.fake.name()
        email = make_au_email(name, self.email_domains)

        return {
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
            "true_segment": cluster_name,
        }

    def generate(self, n_member: int) -> list:
        data = []
        for i in range(n_member):
            data.append(self.generate_member())
            if (i + 1) % max(1, n_member // 10) == 0:
                logger.debug(f"Generated {i + 1}/{n_member} members...")
        return data


def write_members_to_parquet(data: list, output_file: str):
    if not data:
        raise ValueError("No data to write.")
    # Convert list of dicts to dict of lists
    columns = {k: [row[k] for row in data] for k in data[0]}
    table = pa.table(columns)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    logger.info(f"Writing {len(data)} rows to {output_file} using DuckDB...")
    con.register("members", table)
    con.execute(f"COPY members TO '{output_file}' (FORMAT PARQUET)")
    logger.success(f"Generated {len(data)} members and saved to {output_file}")


def main():
    config = ProjectConfig()
    # All config access now via .get(..., file="generate")
    n_row = config.get("data", "n_member", file="generate")
    seed = config.get("data", "random_seed", file="generate")
    output_file = config.get("data", "output_file", file="generate")
    force_generate = config.get("data", "force_generate", file="generate")

    logger.info(
        f"Requested n_member={n_row}, output_file={output_file}, force_generate={force_generate}"
    )

    existing_rows = count_parquet_rows(output_file)
    if existing_rows >= n_row and not force_generate:
        logger.info(
            f"{output_file} already exists with {existing_rows} rows (>= {n_row}); skipping generation."
        )
        return
    elif existing_rows > 0:
        logger.info(
            f"{output_file} exists but has only {existing_rows} rows (< {n_row}); regenerating."
        )

    Faker.seed(seed)
    np.random.seed(seed)

    generator = MemberDataGenerator(config)
    data = generator.generate(n_row)
    write_members_to_parquet(data, output_file)


if __name__ == "__main__":
    main()
