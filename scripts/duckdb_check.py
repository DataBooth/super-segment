from pathlib import Path
import duckdb
import pandas as pd

# Define the database path using pathlib
db_path = Path("data/member_data.duckdb")

# Ensure the parent directory exists
db_path.parent.mkdir(parents=True, exist_ok=True)

# Connect to DuckDB and create a simple table
conn = duckdb.connect(str(db_path))
df = pd.DataFrame({"a": [1, 2, 3]})
conn.execute("CREATE TABLE IF NOT EXISTS test AS SELECT * FROM df")
conn.close()

print("Working directory:", Path.cwd())
print("Database absolute path:", db_path.resolve())
print("Database created:", db_path.exists())
