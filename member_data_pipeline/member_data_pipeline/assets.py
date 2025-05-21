from pathlib import Path

import duckdb
import pandas as pd
from dagster import AssetExecutionContext, asset
from hydra import compose, initialize
from omegaconf import OmegaConf

from super_segment.data_generation import MemberDataGenerator
from super_segment.model import SuperannuationSegmentationModel

DB_PATH = Path(__file__).parent.parent.parent / "data" / "member_data.duckdb"
TABLE = "members"


@asset
def generate_members(context: AssetExecutionContext) -> int:
    """Generate synthetic data and store in DuckDB. Run manually/on demand."""
    # Initialize Hydra and load config
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config")

    # Convert OmegaConf to dict if needed
    config = OmegaConf.to_container(cfg, resolve=True)

    generator = MemberDataGenerator(config)
    df = generator.generate(n_member=config["data"]["n_member"])
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"DROP TABLE IF EXISTS {TABLE}")
    con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM df")
    con.close()
    context.log.info(
        f"Generated and stored {len(df)} members in DuckDB at {DB_PATH.resolve()}"
    )
    return len(df)


@asset
def members_df() -> pd.DataFrame:
    """Load all members from DuckDB."""
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {TABLE}").df()
    con.close()
    return df


@asset
def active_members(members_df: pd.DataFrame) -> pd.DataFrame:
    return members_df[members_df["last_login_days"] < 30]


@asset
def region_counts(active_members: pd.DataFrame) -> pd.DataFrame:
    return (
        active_members.groupby("region")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


@asset
def segmentation_model(active_members: pd.DataFrame) -> dict:
    features = [
        "age",
        "balance",
        "num_accounts",
        "last_login_days",
        "satisfaction_score",
    ]
    X = active_members[features]
    model = SuperannuationSegmentationModel(n_clusters=3)
    stats = model.train(active_members)
    return {"model": model, "stats": stats, "n_samples": len(X)}
