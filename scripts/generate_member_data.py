import os
from pathlib import Path

import duckdb
import hydra
import numpy as np
import pyarrow as pa
from faker import Faker
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def make_au_email(name, domains):
    username = name.lower().replace(" ", ".").replace("'", "").replace("-", "")
    domain = np.random.choice(domains)
    return f"{username}@{domain}"


def count_parquet_rows(parquet_path):
    if not os.path.exists(parquet_path):
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    n_row = config["data"]["n_member"]
    seed = config["data"].get("random_seed", 42)
    output_file = config["data"].get("output_file", "data/members.parquet")
    force_generate = config["data"].get("force_generate", False)

    logger.info(
        f"Requested n_member={n_row}, output_file={output_file}, force_generate={force_generate}"
    )

    # Check for existing file and row count
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
    fake = Faker()

    logger.info("Starting member data generation...")

    # Preallocate arrays/lists
    names = [None] * n_row
    emails = [None] * n_row
    ages = np.empty(n_row, dtype=int)
    balances = np.empty(n_row, dtype=float)
    num_accounts = np.empty(n_row, dtype=int)
    last_login_days = np.empty(n_row, dtype=int)
    satisfaction_scores = np.empty(n_row, dtype=int)
    professions = [None] * n_row
    phases = [None] * n_row
    genders = [None] * n_row
    regions = [None] * n_row
    risk_profiles = [None] * n_row
    contrib_freqs = [None] * n_row
    logins_per_month = np.empty(n_row, dtype=int)
    true_segments = [None] * n_row

    cluster_names = config["clusters"]["segment_names"]
    cluster_probs = config["clusters"]["segment_probs"]
    clusters = config["clusters"]
    email_domains = config["email"]["domains"]

    for i in range(n_row):
        # Pick cluster
        cluster_name = np.random.choice(cluster_names, p=cluster_probs)
        cluster_config = clusters[cluster_name]

        # Generate features
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
        n_acc = np.random.choice(
            cluster_config["num_accounts"]["choices"],
            p=cluster_config["num_accounts"]["probabilities"],
        )
        last_login = int(
            np.clip(
                np.random.exponential(scale=config["last_login_days"]["scale"]),
                config["last_login_days"]["min"],
                config["last_login_days"]["max"],
            )
        )
        satisfaction = np.random.choice(
            config["satisfaction_score"]["choices"],
            p=config["satisfaction_score"]["probabilities"],
        )
        profession = np.random.choice(
            cluster_config["profession"]["choices"],
            p=cluster_config["profession"]["probabilities"],
        )
        phase = np.random.choice(
            config["phase"]["choices"], p=config["phase"]["probabilities"]
        )
        gender = np.random.choice(
            config["gender"]["choices"], p=config["gender"]["probabilities"]
        )
        region = np.random.choice(
            config["region"]["choices"], p=config["region"]["probabilities"]
        )
        risk_profile = np.random.choice(
            cluster_config["risk_profile"]["choices"],
            p=cluster_config["risk_profile"]["probabilities"],
        )
        contrib_freq = np.random.choice(
            config["contrib_freq"]["choices"], p=config["contrib_freq"]["probabilities"]
        )
        logins_pm = int(
            np.clip(
                np.random.poisson(lam=config["logins_per_month"]["mean"]),
                config["logins_per_month"]["min"],
                config["logins_per_month"]["max"],
            )
        )

        name = fake.name()
        email = make_au_email(name, email_domains)

        # Assign
        names[i] = name
        emails[i] = email
        ages[i] = age
        balances[i] = balance
        num_accounts[i] = n_acc
        last_login_days[i] = last_login
        satisfaction_scores[i] = satisfaction
        professions[i] = profession
        phases[i] = phase
        genders[i] = gender
        regions[i] = region
        risk_profiles[i] = risk_profile
        contrib_freqs[i] = contrib_freq
        logins_per_month[i] = logins_pm
        true_segments[i] = cluster_name

        if (i + 1) % max(1, n_row // 10) == 0:
            logger.info(f"Generated {i + 1}/{n_row} members...")

    # Assemble dict-of-arrays for DuckDB
    data = dict(
        name=names,
        email=emails,
        age=ages,
        balance=balances,
        num_accounts=num_accounts,
        last_login_days=last_login_days,
        satisfaction_score=satisfaction_scores,
        profession=professions,
        phase=phases,
        gender=genders,
        region=regions,
        risk_profile=risk_profiles,
        contrib_freq=contrib_freqs,
        logins_per_month=logins_per_month,
        true_segment=true_segments,
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    logger.info(f"Writing {n_row} rows to {output_file} using DuckDB...")

    table = pa.table(data)
    con = duckdb.connect()
    con.register("members", table)
    con.execute(f"COPY members TO '{output_file}' (FORMAT PARQUET)")
    logger.success(f"Generated {n_row} members and saved to {output_file}")


if __name__ == "__main__":
    main()
