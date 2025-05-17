# Model Summary

## 1. **Logistic Regression Model**

The model predicts the probability that a customer will churn (`churned = 1`) based on their features.  
This is a **binary logistic regression**.

### **Model Equation**

Let:

- $x_1$: Number of accounts
- $x_2$: Days since last login
- $x_3$: Satisfaction score
- $x_4$: Age
- $x_5$: Balance

The **logit** (linear combination of features) is:

$$
\text{logit}(p) = \beta_0 + \beta_1 (x_1 - 1) + \beta_2 (x_2 - 90) + \beta_3 (x_3 - 3) + \beta_4 (x_4 - 40) + \beta_5 (x_5 - 50000)
$$

Where the coefficients (from your synthetic data generator) are:

- $\beta_0 = 0 $$ (implicit, or can be added as an intercept)
- $\beta_1 = 0.8 $$                                # effect of more accounts
- $\beta_2 = 0.01 $$                              # effect of less engagement
- $\beta_3 = -0.7 $$                              # higher satisfaction reduces churn
- $\beta_4 = 0.015 $$                             # older age increases churn
- $\beta_5 = 0.000005 $$                          # higher balance increases churn

So, plugging in the values:

$$
\text{logit}(p) = 0.8(x_1 - 1) + 0.01(x_2 - 90) - 0.7(x_3 - 3) + 0.015(x_4 - 40) + 0.000005(x_5 - 50000)
$$

### **Probability Calculation**

The probability of churn is then:

$$
p = \frac{1}{1 + e^{-\text{logit}(p)}}
$$

### **Scaling**

To match the target churn rate (~19%), the probability is scaled:

$$
p_{\text{final}} = p \times 0.28
$$

### **Sampling**

The churn label is generated as:

$$
\text{churned} = 
\begin{cases}
1 & \text{if } u < p_{\text{final}} \\
0 & \text{otherwise}
\end{cases}
$$
where $u \sim \text{Uniform}(0, 1)$.

---

## 2. **Summary Table of Coefficients**

| Feature                | Symbol | Centered at | Coefficient | Effect on Churn    |
|------------------------|--------|-------------|-------------|--------------------|
| Number of accounts     | $x_1$| 1           | 0.8         | More accounts ↑    |
| Days since last login  | $x_2$| 90          | 0.01        | Longer gap ↑       |
| Satisfaction score     | $x_3$| 3           | -0.7        | Higher score ↓     |
| Age                    | $x_4$| 40          | 0.015       | Older ↑            |
| Balance                | $x_5$| 50,000      | 0.000005    | Higher balance ↑   |

---

## 3. **Full Formula**

$$
\boxed{
p_{\text{final}} = 0.28 \times \frac{1}{1 + \exp\left(-\left[0.8(x_1-1) + 0.01(x_2-90) - 0.7(x_3-3) + 0.015(x_4-40) + 0.000005(x_5-50000)\right]\right)}
}
$$

---

## 4. **Interpretation**

- Each feature is centered (subtracting a typical value) to make coefficients interpretable.
- Positive coefficients increase churn probability; negative coefficients decrease it.
- The scaling factor (0.28) is empirical, to match a realistic overall churn rate.