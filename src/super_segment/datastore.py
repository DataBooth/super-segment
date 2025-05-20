import duckdb
import pandas as pd
from pathlib import Path


def get_or_generate_member_data(n_member, generator, config):
    db_path = config["data"]["member_data_db_path"]
    table = config["data"]["member_data_table"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    tables = conn.execute("SHOW TABLES").fetchall()
    if (table,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count >= n_member:
            df = conn.execute(
                f"SELECT * FROM {table} USING SAMPLE {n_member} ROWS"
            ).df()
            conn.close()
            return df
    df = generator.generate(n_member=n_member)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    conn.close()
    return df
