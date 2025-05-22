# Variables
app_name := "app/main.py"
image_name := "super-segment-app"
dockerfile := "Dockerfile"
context := "."

# Default recipe: list all available recipes
default:
    @just --list

# Run the Streamlit app locally (requires activated venv)
app:
    #!/bin/bash
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Please activate the virtual environment first."
        echo "Run 'source .venv/bin/activate' to activate the virtual environment."
        exit 1
    fi
    if ! command -v streamlit >/dev/null 2>&1; then
        echo "Streamlit is not installed in the current environment."
        echo "Run 'uv add streamlit' after activating your virtual environment."
        exit 1
    fi
    streamlit run {{app_name}}

# Sync dependencies using uv
sync:
    uv sync

# Export and update packages from project to requirements.txt
reqs:
    uv export -U > requirements.txt

# --- Docker Recipes ---

# Check Docker context using your tool before building
docker-check:
    docker-context-tree --context {{context}} --dockerfile {{dockerfile}}

# Build the Docker image
docker-build:
    docker build -t {{image_name}} -f {{dockerfile}} {{context}}

# Run the Docker container (adjust ports as needed)
docker-run:
    docker run --rm -p 8501:8501 {{image_name}}

# Clean up dangling Docker images
docker-clean:
    docker image prune -f

# Push Docker image to registry (set $TAG and $REGISTRY as needed)
docker-push tag="latest" registry="":
    #!/bin/bash
    if [ -z "{{registry}}" ]; then
        echo "No registry specified. Skipping push."
        exit 1
    fi
    docker tag {{image_name}} {{registry}}/{{image_name}}:{{tag}}
    docker push {{registry}}/{{image_name}}:{{tag}}


# Generate member data
# just generate-member-data n_member=10000 force_generate=true
generate-member-data n_member="10000" force_generate="false":
    python scripts/generate_member_data.py data.n_member={{n_member}} data.force_generate={{force_generate}}

    
# Transform and query with Ibis/DuckDB
transform-members:
    python scripts/transform_members.py

# Full pipeline
pipeline:
    just generate-members
    just transform-members


# -------- Dagster ---------- NOT WORKING

# # Generate member data (on demand)
# generate-members:
#     dagster asset materialize -m member_data_pipeline.member_data_pipeline --select generate_members


# # Run the full pipeline (materialize all assets except generate_members) - "+" means downstream of members_df
# run-pipeline:
#     dagster asset materialize --select members_df+  

# # Optionally, run everything including generate_members (rarely needed)
# full-refresh:
#     dagster asset materialize --select generate_members,members_df+

# # Open Dagster UI for interactive runs
# dagster-ui:
#     dagster dev

# # Show pipeline status in Dagster UI
# status:
#     dagster job list
