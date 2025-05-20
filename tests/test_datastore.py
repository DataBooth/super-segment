import pytest
import os
import duckdb
from super_segment.data_generation import (
    get_or_generate_member_data,
    MemberDataGenerator,
)


@pytest.fixture
def db_config(tmp_path, config):
    db_path = tmp_path / "test.duckdb"
    config["data"]["member_data_db_path"] = str(db_path)
    config["data"]["member_data_table"] = "members"
    return config


def test_duckdb_caching(db_config):
    gen = MemberDataGenerator(db_config)
    df1 = get_or_generate_member_data(5, gen, db_config)
    assert len(df1) == 5
    # Should now load from DB, not regenerate
    df2 = get_or_generate_member_data(5, gen, db_config)
    assert len(df2) == 5
    # DB file should exist
    assert os.path.exists(db_config["data"]["member_data_db_path"])
    # Table should exist
    conn = duckdb.connect(db_config["data"]["member_data_db_path"])
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    assert db_config["data"]["member_data_table"] in tables
    conn.close()
