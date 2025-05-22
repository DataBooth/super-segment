FROM python:3.12-slim-bullseye

# Install uv
RUN pip install -U uv

WORKDIR /app

# Create a non-root user
RUN useradd -m appuser

# Copy pyproject.toml first for better cache
COPY pyproject.toml pyproject.toml

# Create the venv as root so permissions are correct
RUN uv venv .venv

# Set env vars for the venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Change ownership of /app to appuser so they can write files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Install dependencies (from lock file)
RUN uv sync --no-dev

# Copy application code, config, and data (as appuser for correct permissions)
COPY --chown=appuser:appuser app app
COPY --chown=appuser:appuser conf conf
COPY --chown=appuser:appuser data data
COPY --chown=appuser:appuser README.md .

# Install your local package (editable or normal)
# RUN uv pip install -e . (already happened in uv sync)

# EXPOSE 8501

CMD sh -c 'streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0'

