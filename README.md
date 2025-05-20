# Superannuation Member Segmentation Project

## Overview

This project provides a comprehensive framework for segmenting superannuation fund members using both rule-based and machine learning approaches. It includes synthetic data generation, model development, and an interactive Streamlit web application for exploring and visualising segmentation results. The aim is to help super funds better understand, engage, and serve their members—whether in the accumulation or retirement phase—while upholding ethical and legal standards.

---

## Objectives

- Identify and analyse meaningful member segments to support targeted communications, product recommendations, and proactive member support.
- Compare rule-based and algorithmic segmentation methods, using clear evaluation metrics.
- Provide an interactive interface for fund analysts and stakeholders to explore segments, test new member profiles, and visualise key patterns.
- Ensure compliance with Australian Privacy Principles (APPs) and ethical best practices throughout.

---

## Key Features

- **Synthetic Data Generation:**  
  Create semi-realistic member datasets including demographic, behavioural, and psychographic features relevant to superannuation.
- **Multiple Segmentation Approaches:**  
  - Rule-based (e.g., by age, phase, balance)
  - Model-based (e.g., K-Means, hierarchical clustering)
- **Evaluation Metrics:**  
  Silhouette score, cluster interpretability, and business actionability.
- **Streamlit App:**  
  - Visualise segment profiles and distributions
  - Compare segmentation methods
  - Profile new or hypothetical members
  - Download segment summaries for reporting
- **Ethical and Legal Compliance:**  
  Designed to align with the Australian Privacy Principles and industry ethical standards.

---

## Folder Structure

```
superannuation-segmentation/
│
├── data/
│   └── synthetic_members.csv
├── src/
│   ├── data_generation.py
│   ├── segmentation_models.py
│   ├── evaluation.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── README.md
└── requirements.txt
```

---

## Data Description

Synthetic member data includes:
- Age, gender, profession (e.g., high school teacher, admin)
- Account balance, number of accounts
- Super phase (accumulation or retirement)
- Contribution frequency, satisfaction score, risk profile
- Digital engagement (logins, last login days)
- Region (state/territory)

All data is generated to reflect plausible distributions and correlations typical in the Australian superannuation sector.

---

## Modelling Approaches

### Rule-Based Segmentation

- Define segments using business logic (e.g., age bands, phase, balance quartiles).
- Useful for transparency and alignment with fund strategy.

### Model-Based Segmentation

- **K-Means Clustering:** Finds natural groupings in the data based on selected features.
- **Hierarchical Clustering:** Builds a tree of segments for more granular analysis.
- **Gaussian Mixture Models (optional):** For probabilistic/soft segmentation.

### Evaluation

- **Silhouette Score:** Measures how well-defined the clusters are.
- **Interpretability:** Are segments meaningful and actionable for the fund?
- **Business Usefulness:** Can segments support specific communications or interventions?

---

## Streamlit Application

- **Segment Explorer:**  
  Visualise clusters, segment sizes, and average characteristics.
- **Method Comparison:**  
  Switch between rule-based and model-based segmentations.
- **Member Profiler:**  
  Enter or simulate member details to see predicted segment and suggested actions.
- **Download Reports:**  
  Export segment summaries for further analysis.

---

## Ethical and Legal Considerations

- **Australian Privacy Principles (APPs):**  
  Data collection, processing, and use are designed to comply with APPs, especially regarding transparency, consent, data minimisation, and member access rights.
- **Fairness and Non-Discrimination:**  
  Segmentation avoids reinforcing stereotypes or unfairly excluding groups. Regular bias checks are recommended.
- **Transparency:**  
  The app provides clear explanations of how segmentation works, why members are grouped as they are, and what actions may result.
- **Data Quality:**  
  Recognises that behavioural data (e.g., digital engagement) may be sparse for many members. Models are designed to avoid overfitting or overinterpreting limited signals.
- **Inclusivity:**  
  Segmentation is designed to benefit all members, including those with low engagement or in vulnerable situations.

For more detail on ethical considerations and APP compliance, see the [Ethics and Privacy](#ethics-and-privacy) section below.

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip (for dependencies)
- Streamlit

### Installation

```bash
git clone https://github.com/your-org/superannuation-segmentation.git
cd superannuation-segmentation
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app/streamlit_app.py
```

---

## Customisation

- Adjust synthetic data parameters in `src/data_generation.py` to better reflect your fund’s member base.
- Add or remove features in `segmentation_models.py` as needed.
- Update business rules for rule-based segmentation in `segmentation_models.py`.
- Extend the Streamlit interface in `app/streamlit_app.py` for additional visualisations or export options.

---

## Ethics and Privacy

This project is designed with strict adherence to the Australian Privacy Principles (APPs):

- **Open and Transparent Management (APP 1):**  
  The app includes a clear privacy notice and documentation.
- **Collection and Notification (APPs 3 & 5):**  
  Only necessary data is generated and used, with clear documentation on purpose.
- **Use and Disclosure (APP 6):**  
  Data is used solely for segmentation and member support purposes.
- **Security (APP 11):**  
  Synthetic data is used for demonstration; in production, ensure robust security for member data.
- **Access and Correction (APP 12):**  
  Members (or test users) can review and correct their data in the app interface.

**Note:** When adapting for real member data, ensure all privacy, consent, and ethical requirements are fully met.

---

## References

- [The psychology of super fund member segmentation][6]
- [Australian Privacy Principles – OAIC](https://www.oaic.gov.au/privacy/australian-privacy-principles)
- [Customer segmentation templates and guides][3][4][5]

---

## Contact

For questions, suggestions, or contributions, please raise an issue or contact the project maintainer.

---

*This project is for educational and demonstration purposes.*

---

[6]: https://www.fssuper.com.au/article/the-psychology-of-super-fund-member-segmentation  
[3]: https://www.slideteam.net/blog/top-10-customer-segmentation-templates-with-samples-and-examples  
[4]: https://boardmix.com/articles/market-segmentation-example/  
[5]: https://www.ayoa.com/templates/market-segmentation-template/

Citations:
[1] https://github.com/stevehoober254/customer_segmentation/blob/main/README.md
[2] https://github.com/Suwarti/Customer-Segmentation/blob/main/README.md
[3] https://www.slideteam.net/blog/top-10-customer-segmentation-templates-with-samples-and-examples
[4] https://boardmix.com/articles/market-segmentation-example/
[5] https://www.ayoa.com/templates/market-segmentation-template/
[6] https://www.fssuper.com.au/article/the-psychology-of-super-fund-member-segmentation
[7] https://www.youtube.com/watch?v=SRr0tlUJjcw
