import marimo as mo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import (
    MemberDataGenerator,
    SuperannuationChurnModel,
    generate_age,
    generate_balance,
    generate_num_accounts,
    generate_last_login_days,
    generate_satisfaction_score,
    compute_churn_probability,
)

# Load config (adapt path as needed)
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Generate member data
gen = MemberDataGenerator(config)
df = gen.generate(1000)

# Model training
model = SuperannuationChurnModel()
stats = model.train(df)

# 1. Show class balance
sns.countplot(x="churned", data=df)
plt.title("Churn Class Balance")
plt.show()

# 2. Show feature distributions
for col in ["age", "balance", "num_accounts", "last_login_days", "satisfaction_score"]:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 3. Correlation heatmap
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# 4. Model performance metrics
print("Model Accuracy:", stats["accuracy"])
print("Classification Report:")
print(pd.DataFrame(stats["report"]).transpose())

# 5. Sensitivity: Varying satisfaction_score
ages = [30, 40, 50, 60]
sats = np.arange(1, 6)
results = []
for age in ages:
    for sat in sats:
        prob = compute_churn_probability(
            num_accounts=1,
            last_login_days=30,
            satisfaction_score=sat,
            age=age,
            balance=50000,
            config=config,
        )
        results.append({"age": age, "satisfaction_score": sat, "churn_prob": prob})
df_sens = pd.DataFrame(results)
sns.lineplot(data=df_sens, x="satisfaction_score", y="churn_prob", hue="age")
plt.title("Churn Probability vs Satisfaction Score (by Age)")
plt.show()
