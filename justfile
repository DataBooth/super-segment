
# Default recipe: list all available recipes
default:
    @just --list

# Run the Streamlit app
app:
    streamlit run app/main.py


# Generate synthetic member data
generate-member-data n_member="1000":
    python scripts/generate_member_data.py 
    
# Transform and query with Ibis/DuckDB
transform-members:
    python scripts/transform_members.py

# Full pipeline
pipeline:
    just generate-members
    just transform-members


# -------- Dagster ----------
# Generate member data (on demand)
generate-members:
    dagster asset materialize -m member_data_pipeline.member_data_pipeline --select generate_members


# Run the full pipeline (materialize all assets except generate_members) - "+" means downstream of members_df
run-pipeline:
    dagster asset materialize --select members_df+  

# Optionally, run everything including generate_members (rarely needed)
full-refresh:
    dagster asset materialize --select generate_members,members_df+

# Open Dagster UI for interactive runs
dagster-ui:
    dagster dev

# Show pipeline status in Dagster UI
status:
    dagster job list
