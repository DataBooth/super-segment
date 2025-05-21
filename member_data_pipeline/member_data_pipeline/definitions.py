from dagster import Definitions
from .assets import generate_members, members_df, active_members, region_counts, segmentation_model

defs = Definitions(
    assets=[
        generate_members,
        members_df,
        active_members,
        region_counts,
        segmentation_model,
    ]
)
