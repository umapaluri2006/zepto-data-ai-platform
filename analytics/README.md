# Module 2 — Analytics Pipeline (/analytics)

## Overview & Architecture
This module represents the complete Analyst-to-Data-Scientist workflow for Zepto: loading and profiling customer/passenger data once, handling missing values defensibly, engineering features, uncovering demographic survival patterns, training and rigorously evaluating 3 classification pipelines with imbalance handling and hyperparameter tuning, conducting a multivariate fare regression with residual analysis, and deploying a single bundled pipeline artifact.

```
                  titanic.csv (Committed Offline Fallback)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        [01_eda.ipynb (Part A)]               [02_modeling.ipynb (Part B)]
        ├── Profiling & Info                  ├── Stratified Train/Test Split (80/20)
        ├── Threshold Missing Handling        ├── ColumnTransformer (Fit on Train Only)
        ├── IQR Outliers & Skewness           ├── 3 Classifiers (LogReg, DT, RF)
        ├── Boolean Masking Bivariate         ├── SMOTE vs class_weight Comparison
        ├── 6x6 Correlation Heatmap           ├── GridSearchCV (oob_score=True)
        ├── 4-Chart Multivariate Story        ├── Fare Regression & Heteroscedasticity
        └── Exploratory Z-Score Check         └── joblib Pipeline Export & Reload Verification
```

---

## Part A: Profiling, Cleaning & Exploratory Data Analysis

### 1. Dataset Profiling & Missing-Value Threshold Strategy
- **Total Records**: 891 rows, 15 columns.
- **Missing Value Percentages**:
  - `deck`: **77.22%** (688 / 891)
  - `age`: **19.87%** (177 / 891)
  - `embarked` / `embark_town`: **0.22%** (2 / 891)
- **Threshold Rule Implementation**:
  - `< 5% missing`: `embarked` dropped for rows (or mode imputed) due to minimal sample loss.
  - `5% – 30% missing`: `age` imputed with **median** (28.0) to preserve statistical distribution and prevent data shrinkage.
  - `> 70% missing`: `deck` dropped entirely because imputing $>77\%$ missing values would inject synthetic noise and invalidate statistical inference.
  - `alive`, `embark_town`, `class`: Dropped as redundant duplicates of `survived`, `embarked`, and `pclass`.

### 2. Univariate Analysis & Outlier Detection
Using the Interquartile Range (IQR) rule ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$):
- **Age**: $Q_1 = 22.0, Q_3 = 35.0, \text{IQR} = 13.0 \implies [2.50, 54.50]$. Outliers: **65 rows (7.31%)**.
- **Fare**: $Q_1 = 7.90, Q_3 = 31.00, \text{IQR} = 23.10 \implies [-26.76, 65.66]$. Outliers: **114 rows (12.82%)**.
- **Fare Distribution Skewness**:
  $$\text{Mean } (\$32.10) > \text{Median } (\$14.45) > \text{Mode } (\$8.05)$$
  Because the mean is pulled heavily to the right of both median and mode by luxury 1st-class tickets, the fare distribution is **strongly right-skewed (positive skew)**.

### 3. Bivariate Analysis: Boolean Masking & 6x6 Correlation Matrix
- **Survival Rate by Demographics**:
  - **Sex**: Female = **74.04%**, Male = **18.89%**.
  - **Class**: 1st Class = **62.62%**, 2nd Class = **47.28%**, 3rd Class = **24.24%**.
  - **Sex + Class Interaction**:
    - Female Class 1: **96.74%** | Female Class 2: **92.11%** | Female Class 3: **50.00%**
    - Male Class 1: **36.89%** | Male Class 2: **15.74%** | Male Class 3: **13.54%**
- **6x6 Correlation Matrix (Excluding redundant `adult_male` and `alone`)**:
  - **Top 1 Pair**: `(pclass, fare)` with Pearson $r = -0.548$ ($|r| = 0.548$). Strong negative correlation reflecting socioeconomic ticket pricing — higher tier cabins (1st class) cost dramatically more.
  - **Top 2 Pair**: `(sibsp, parch)` with Pearson $r = +0.415$ ($|r| = 0.415$). Moderate positive correlation reflecting family units — passengers traveling with siblings/spouses were also much more likely to travel with parents/children.

### 4. Multivariate Data Story (4 Visual Arguments)
1. **Chart 1 (Pclass $\times$ Sex $\rightarrow$ Survival)**: Demonstrates the compounding effect of gender and social class; 1st-class women were virtually guaranteed rescue ($96.7\%$), whereas 3rd-class men faced extreme casualty rates ($13.5\%$).
2. **Chart 2 (Age $\times$ Pclass $\rightarrow$ Survival)**: Shows child prioritization was effective across classes, but adult survival was predominantly determined by cabin class.
3. **Chart 3 (Fare vs Age $\times$ Survival)**: Confirms higher fare ticket holders clustered heavily among survivors.
4. **Chart 4 (Family Size $\rightarrow$ Survival)**: Moderate family sizes (2–4 members) achieved optimal survival ($\sim 55\%-70\%$), while solo travelers and very large families ($>5$) suffered due to isolation or evacuation coordination breakdown.

### 5. Exploratory Z-Score Standardization Sanity Check
- $z = \frac{x - \mu}{\sigma}$ transformed `age` and `fare` on full cleaned data:
  - Age: Mean $= 0.0000$, Std Dev $= 1.0000$ (Min: $-2.2253$, Max: $3.9034$)
  - Fare: Mean $= 0.0000$, Std Dev $= 1.0000$ (Min: $-0.6458$, Max: $9.6631$)

---

## Part B: Predictive Modeling & Pipeline Deployment

### 1. Stratified Train/Test Split
- **Split**: 80% Train ($n=712$), 20% Test ($n=179$), `stratify=y`, `random_state=42`.
- **Justification**: Target `survived` has a 61.6% (Died) to 38.4% (Survived) class balance. Stratification ensures both partitions reflect identical target distributions, avoiding biased test evaluations.

### 2. Preprocessing Architecture (`ColumnTransformer`)
- Structurally enforced **fit on train only**, transform on test:
  - `numeric`: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
  - `categorical`: `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(drop='first', handle_unknown='ignore')`

### 3. Classification Models & Imbalance Handling
- Evaluated **Logistic Regression**, **Decision Tree (max_depth=4)**, and **Random Forest**.
- **Imbalance Handling Comparison (Random Forest)**:
  - **Baseline**: Precision $= 0.8077$, Recall $= 0.6087$, F1 $= 0.6942$
  - **`class_weight='balanced'`**: Precision $= 0.7581$, Recall $= 0.6812$, F1 $= 0.7176$
  - **`SMOTE` (Train Only)**: Precision $= 0.7463$, Recall $= 0.7246$, F1 $= 0.7353$
- **Conclusion**: For rescue and passenger safety operations where false negatives (failing to predict a survivor) are critical, `SMOTE` and `class_weight='balanced'` provide superior recall improvements.

### 4. Hyperparameter Tuning via GridSearchCV
- Performed on `RandomForestClassifier(oob_score=True, random_state=42)` across `n_estimators`, `max_depth`, `max_features` with 5-fold Stratified CV.
- **Best Parameters**: `{'classifier__max_depth': 4, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 150}`
- **Out-of-Bag (OOB) Score**: **0.8301**
- **Test Set Metrics**: Accuracy $= 0.7989$, Precision $= 0.8667$, Recall $= 0.5652$, F1 $= 0.6842$, AUC $= 0.8404$.

### 5. Regression Side-Task: Predicting Fare
- Multivariate Linear Regression using demographic & ticket features.
- **Metrics**:
  - $\text{MAE} = \$18.87$
  - $\text{RMSE} = \$30.92$
  - $R^2 = 0.3822$
  - $\text{Adjusted } R^2 = 0.3531$
- **Heteroscedasticity Analysis**: The residual scatter plot ($\hat{y}$ vs $y - \hat{y}$) shows a clear funnel shape where error variance expands dramatically at higher predicted fares. This confirms strong **heteroscedasticity**, suggesting non-linear or log-transformed models are necessary for ticket pricing.

---

## Master Model Comparison Table

| Model Type | Model / Estimator | Accuracy | Precision | Recall | F1 Score | AUC-ROC | MAE ($) | RMSE ($) | R² | Adj R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classification** | Logistic Regression | **0.7989** | 0.7797 | **0.6667** | **0.7188** | **0.8436** | *N/A* | *N/A* | *N/A* | *N/A* |
| **Classification** | Decision Tree (depth=4) | 0.7933 | 0.8333 | 0.5797 | 0.6838 | 0.8281 | *N/A* | *N/A* | *N/A* | *N/A* |
| **Classification** | Random Forest (Tuned) | **0.7989** | **0.8667** | 0.5652 | 0.6842 | 0.8404 | *N/A* | *N/A* | *N/A* | *N/A* |
| **Regression** | Multivariate Linear Reg (Fare) | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | **18.87** | **30.92** | **0.3822** | **0.3531** |

*(Note: Classification and Regression metrics operate on separate mathematical dimensions and are presented in distinct metric columns above).*

---

## Deployment Recommendation

> **Recommended Model for Production Deployment: Logistic Regression (or Tuned Random Forest with Balanced Weights)**
> 
> For operational deployment in Zepto's analytics environment, **Logistic Regression** is recommended as the primary classification engine because it achieves the highest balanced F1 Score (**0.7188**), highest AUC-ROC (**0.8436**), and best recall (**0.6667**) among baseline classifiers, while providing high inference speed and transparent feature odds ratios. When extreme precision is demanded (avoiding false positive survivor alerts), the **Tuned Random Forest** is the preferred alternative with a top precision of **0.8667** and robust Out-of-Bag generalization score of **0.8301**.

---

## Joblib Pipeline Export & Verification

The best complete pipeline (`ColumnTransformer` + estimator) is saved to `analytics/best_model_pipeline.joblib`.

```python
import joblib, pandas as pd, numpy as np

# Load serialized pipeline
pipeline = joblib.load("analytics/best_model_pipeline.joblib")

# Inference directly on raw, unpreprocessed data (handles missing values automatically)
raw_input = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": np.nan,  # Auto-imputed via pipeline
    "sibsp": 1,
    "parch": 0,
    "fare": 150.0,
    "embarked": "S"
}])

prediction = pipeline.predict(raw_input)
survival_prob = pipeline.predict_proba(raw_input)[:, 1]
print(f"Predicted Class: {prediction[0]}, Survival Probability: {survival_prob[0]:.4f}")
```

**Verification Status**: Tested and confirmed working end-to-end on raw inputs.
