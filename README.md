# Superannuation Churn Predictor

A Streamlit web application that generates synthetic superannuation customer data, trains a churn prediction model, allows interactive predictions, visualises data distributions, and demonstrates model fit quality.


- [Superannuation Churn Predictor](#superannuation-churn-predictor)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Data Documentation](#data-documentation)
    - [Data Generation Process](#data-generation-process)
      - [Example Generation Logic](#example-generation-logic)
    - [Data Schema](#data-schema)
      - [Sample Data](#sample-data)
    - [Data Usage](#data-usage)
    - [Data Limitations](#data-limitations)
    - [Customisation](#customisation)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

---

## Features

- **Synthetic Data Generation:** Uses the [Faker](https://faker.readthedocs.io/) library to create realistic, privacy-safe customer records.
- **Model Training:** Trains a logistic regression model to predict customer churn.
- **Interactive Prediction:** Enter customer features and receive churn probability predictions.
- **Tabbed Interface:** Explore data, train the model, make predictions, visualise distributions, view model diagnostics, and read project documentation.
- **Object-Oriented Design:** All logic is encapsulated in Python classes for maintainability and clarity.
- **Logging:** Uses [Loguru](https://loguru.readthedocs.io/) for clear, informative logs.
- **Caching:** Synthetic data generation is cached for performance.

---

## Getting Started

### Prerequisites

- Python 3.11+
- `uv`


### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/superannuation-churn-predictor.git
cd superannuation-churn-predictor
```

2. **Install dependencies:**

```bash
uv sync
```

See **`pyproject.toml`** which includes:

```
streamlit
pandas
numpy
scikit-learn
faker
loguru
plotly
statsmodels
watchdog
```


---

## Usage

1. **Run the Streamlit app:**

```bash
streamlit run app/main.py
```

2. **Navigate the app:**
    - **Data Sample Tab:** Preview a sample of the generated synthetic data.
    - **Model Training Tab:** Train the churn prediction model and review its accuracy and classification metrics.
    - **Predict Churn Tab:** Input customer features and get a churn probability and prediction.
    - **Data Distributions Tab:** Explore distributions of all numerical features.
    - **Model Fit Visualisation Tab:** See residual and actual-vs-predicted plots to assess model fit.
    - **Readme Tab:** View this documentation within the app.

---

## Data Documentation

### Data Generation Process

- **Library:** [Faker](https://faker.readthedocs.io/) is used to create realistic names and emails.
- **Randomisation:** `NumPy` is used to generate numerical features within realistic ranges for Australian superannuation members.
- **Churn Label:** The `churned` field is assigned based on a simple rule, simulating members likely to leave the fund.


#### Example Generation Logic

```python
age = np.random.randint(25, 66)
balance = np.random.randint(20000, 300001)
num_accounts = np.random.randint(1, 5)
last_login_days = np.random.randint(1, 181)
satisfaction_score = np.random.randint(1, 6)
churned = int((num_accounts > 2) or (last_login_days > 60) or (satisfaction_score < 3))
```


### Data Schema

| Column | Type | Description |
| :-- | :-- | :-- |
| name | string | Synthetic full name |
| email | string | Synthetic email address |
| age | int | Age of the member (25-65) |
| balance | int | Superannuation account balance (AUD 20,000–300,000) |
| num_accounts | int | Number of super accounts held (1–4) |
| last_login_days | int | Days since last online account login (1–180) |
| satisfaction_score | int | Last member satisfaction survey score (1–5) |
| churned | int | 1 = member churned (left fund), 0 = remained |

#### Sample Data

| name | email | age | balance | num_accounts | last_login_days | satisfaction_score | churned |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| John Smith | john.smith@fake.com | 35 | 55000 | 2 | 30 | 3 | 0 |
| Jane Doe | jane.doe@fake.com | 52 | 210000 | 1 | 5 | 5 | 0 |
| Michael Brown | michael.brown@fake.com | 47 | 120000 | 3 | 90 | 2 | 1 |

### Data Usage

- **Model Training:** Used to train machine learning models for churn prediction.
- **Exploration:** Supports data analysis and visualisation in the Streamlit app.
- **Demonstration:** Enables safe, privacy-compliant demonstration of ML workflows.


### Data Limitations

- **Synthetic:** All data is randomly generated and does not reflect real individuals.
- **Simplified Logic:** Churn assignment is based on a basic rule for demonstration, not real-world behavior.
- **No PII Risk:** Names and emails are fake; no real customer data is used.


### Customisation

You can adjust the data generator to:

- Add new features (e.g., employer, location, investment choice).
- Change value ranges for different scenarios.
- Implement more complex churn logic or introduce randomness.

---

## Project Structure

```
superannuation-churn-predictor/
│
├── app.py               # Main Streamlit application
├── README.md            # Project documentation (this file)
├── requirements.txt     # Python dependencies
└── (Optional: src/)     # For additional modules/classes if you modularise further
```


---

## Contributing

Pull requests and suggestions are welcome! Please open an issue or submit a PR.

## License

[MIT License](LICENSE)

## Contact

For questions or support, please contact [michael@databooth.com.au](mailto:michael@databooth.com.au).

---

*This synthetic dataset and app are intended solely for demonstration and educational purposes. They should not be used for production or to model real-world superannuation behaviour without further validation and enhancement.*