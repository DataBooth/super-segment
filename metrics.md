### **Understanding the Classification Report Table**

The table below summarises the performance of the churn prediction model for each class:

| Metric | Description |
| :-- | :-- |
| **precision** | The proportion of positive identifications that were actually correct. |
| **recall** | The proportion of actual positives that were correctly identified. |
| **f1-score** | The harmonic mean of precision and recall (higher is better). |
| **support** | The number of true instances for each class in the test data. |

- **Rows labeled `0` and `1`** correspond to the two classes:
    - `0`: Not churned
    - `1`: Churned
- **Averages** (`macro avg`, `weighted avg`) provide overall model performance summaries.
- **All values are shown to two decimal places for clarity.**

---

**How to interpret:**

- **High precision** for class `1` means most predicted churns are correct.
- **High recall** for class `1` means most actual churns are detected.
- **f1-score** balances both.
- **Support** shows how many samples were in each group.