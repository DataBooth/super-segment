# Model Summary

## 1. **KMeans Clustering Model**

The segmentation model groups members into $K$ distinct segments based on their features, using the **KMeans clustering algorithm**.

---

### **Feature Vector**

Each member is represented as a feature vector:

$$
\mathbf{x} = [x_1, x_2, ..., x_p]
$$

where the features may include, for example:

- $x_1$: Age
- $x_2$: Balance
- $x_3$: Number of accounts
- $x_4$: Days since last login
- $x_5$: Satisfaction score
- $x_6$: Logins per month
- $x_7$ – $x_p$: One-hot encoded categorical features (e.g., profession, phase, gender, region, risk profile, contribution frequency)

---

### **Preprocessing**

- **Numeric features** are standardised (zero mean, unit variance):
  $$
  x_j' = \frac{x_j - \mu_j}{\sigma_j}
  $$
  where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$.

- **Categorical features** are one-hot encoded:
  $$
  \text{e.g., Profession} = \begin{bmatrix}0 & 1 & 0 & 0 & 0\end{bmatrix}
  $$
  for "Primary Teacher" if the categories are ["High School Teacher", "Primary Teacher", ...].

---

### **KMeans Objective**

Given $N$ members and $K$ clusters, KMeans finds cluster centroids $\boldsymbol{\mu}_1, ..., \boldsymbol{\mu}_K$ to minimise the total within-cluster variance:

$$
\text{Objective:} \qquad
\min_{C_1,...,C_K} \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

where $C_k$ is the set of members assigned to cluster $k$ and $\boldsymbol{\mu}_k$ is the centroid (mean vector) of cluster $k$.

---

### **Cluster Assignment**

Each member is assigned to the segment whose centroid is closest (Euclidean distance):

$$
\text{segment}(\mathbf{x}_i) = \arg\min_{k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|
$$

---

## 2. **Model Evaluation**

### **Silhouette Score**

The **silhouette score** measures how well each member fits within its segment compared to other segments:

$$
s = \frac{b - a}{\max(a, b)}
$$

- $a$ = mean intra-cluster distance (distance to members in the same cluster)
- $b$ = mean nearest-cluster distance (distance to members in the nearest different cluster)
- $s$ ranges from -1 (poor fit) to +1 (well clustered).

---

## 3. **Interpretation**

- **Segments** represent groups of members with similar characteristics and behaviours.
- **Cluster centroids** describe the "average" member in each segment.
- **Segment profiles** can be used for targeted communications, product recommendations, and member engagement strategies.

---

## 4. **Summary Table of Features**

| Feature                | Type         | Example Values                | Notes                          |
|------------------------|-------------|-------------------------------|--------------------------------|
| Age                    | Numeric      | 25–65                         | Standardised                   |
| Balance                | Numeric      | \$20,000–\$300,000            | Standardised                   |
| Number of accounts     | Numeric      | 1–4                           | Standardised                   |
| Days since last login  | Numeric      | 1–180                         | Standardised                   |
| Satisfaction score     | Numeric      | 1–5                           | Standardised                   |
| Logins per month       | Numeric      | 0–20                          | Standardised                   |
| Profession             | Categorical  | High School Teacher, etc.     | One-hot encoded                |
| Phase                  | Categorical  | Accumulation, Retirement      | One-hot encoded                |
| Gender                 | Categorical  | Male, Female                  | One-hot encoded                |
| Region                 | Categorical  | NSW, VIC, QLD, WA, SA         | One-hot encoded                |
| Risk profile           | Categorical  | Conservative, Moderate, Aggressive | One-hot encoded          |
| Contribution frequency | Categorical  | Monthly, Quarterly, Yearly    | One-hot encoded                |

---

## 5. **Cluster Visualisation**

- Clusters can be visualised in 2D using PCA or t-SNE projections.
- Each point represents a member, coloured by segment.

---

## 6. **Interpretation for Business**

- Each segment can be profiled by average feature values.
- Segments inform tailored marketing, communications, and product design.
- The number of clusters ($K$) is chosen based on business needs and evaluation metrics (e.g., silhouette score).

## Sensitivity Analysis

## Age Cohort Sensitivity Analysis

To ensure the robustness of our clustering results to the choice of age cohort boundaries, we perform a sensitivity analysis. We systematically vary the boundaries of each age cohort by ±1 and ±2 years and re-run the clustering within each cohort. We then compare the resulting cluster assignments, sizes, and characteristics across these configurations. Stable results indicate that our findings are not artifacts of arbitrary age binning.

## Fuzzy Clustering

In addition to hard clustering, we apply fuzzy clustering (fuzzy c-means), which allows each member to have degrees of membership in multiple clusters. This approach reflects the reality that some individuals may not fit neatly into a single group. We present the membership matrix and visualize the cluster centers to aid interpretation.

---

**Note:**  

Unlike logistic regression, clustering does not yield explicit coefficients or probabilities, but instead finds natural groupings in the data based on all included features.