# Employee Attrition Analytics & Flight-Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%26%20CV-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Viz-4C72B0)](https://seaborn.pydata.org/)
[![Status](https://img.shields.io/badge/Version-v2.0%20Modernized-success)]()

An end-to-end **People Analytics & Machine Learning Case Study** on the **IBM HR Analytics Employee Attrition & Performance** dataset ($N = 1,470$ employees, $35$ attributes).

This repository contains both the original **v1 notebook** (built ~5–6 years ago) and a **modernized v2 refactor** ([`HR_Analytics_Case_Study_v2.ipynb`](./HR_Analytics_Case_Study_v2.ipynb)) that serves as a practical case study in fixing common real-world machine learning pitfalls: the **imbalanced accuracy trap**, **arbitrary label encoding of nominal categories**, **data leakage**, and **raw-count vs. rate fallacies in EDA**.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| **[`HR_Analytics_Case_Study_v2.ipynb`](./HR_Analytics_Case_Study_v2.ipynb)** | **Modernized v2 Notebook (Recommended)**: Normalized rate-based EDA, domain feature engineering, leak-free `ColumnTransformer` + `Pipeline`, 5-fold `StratifiedKFold` cross-validation, OOF threshold tuning, 95% bootstrap CIs, slice-based fairness audit, and directional explainability. |
| [`HR analytics (Case Study).ipynb`](./HR%20analytics%20(Case%20Study).ipynb) | **Legacy v1 Notebook**: Original exploratory analysis and baseline modeling from ~5–6 years ago (preserved for comparison). |
| [`WA_Fn-UseC_-HR-Employee-Attrition.csv`](./WA_Fn-UseC_-HR-Employee-Attrition.csv) | IBM HR Analytics Employee Attrition & Performance dataset ($1,470$ rows $\times$ $35$ columns). |
| [`requirements.txt`](./requirements.txt) | Python dependencies required to reproduce the analysis and modeling pipeline. |
| [`README.md`](./README.md) | Project documentation, methodology comparison (v1 vs. v2), key findings, and retention recommendations. |

---

## Why v2? What Was Fixed from v1

Revisiting an older data science project highlights how crucial rigorous validation and problem framing are. Here is a summary of what went wrong in v1 and how v2 resolves each issue:

| Dimension | v1 Implementation (Legacy) | v2 Implementation (Modernized) |
| :--- | :--- | :--- |
| **1. Imbalanced Evaluation Metric** | Evaluated models solely on `.score()` (**Accuracy**). Because **83.88%** of employees stay (`No`), a naive dummy classifier gets **83.88% accuracy** while catching **0% of leavers**. | Evaluates **PR-AUC (Average Precision)**, **ROC-AUC**, **Recall**, **F1-Score**, and **Confusion Matrices** with **95% Bootstrap Confidence Intervals**, plus out-of-fold probability threshold tuning. |
| **2. Categorical Encoding** | Manually mapped nominal strings (`JobRole`, `Department`, `EducationField`, `MaritalStatus`) to arbitrary integers (`0..8`, skipping `2` in `MaritalStatus`), imposing a false numerical hierarchy (`Divorced = 3 × Married`). | Uses `OneHotEncoder(drop='first')` for nominal categories, `OrdinalEncoder` for `BusinessTravel` (`Non-Travel < Travel_Rarely < Travel_Frequently`), and preserves pre-encoded 1–4/1–5 survey scales. |
| **3. Feature Scaling** | Fed unscaled `MonthlyIncome` (~`$20,000`) and `DailyRate` (~`$1,500`) alongside `0/1` flags into L2-regularized `LogisticRegression`, causing poor convergence (`83.33%` accuracy — worse than guessing all `0`s). | Standardizes all continuous and ordinal features via `StandardScaler` inside a leak-free `Pipeline`, making `LogisticRegression` the top-performing model (`0.838` CV ROC-AUC, `0.668` tuned CV PR-AUC). |
| **4. Data Leakage & Feature Selection** | Encoded full DataFrame before `train_test_split` (non-stratified); selected 11 features via `SelectFromModel` in Cell 25 but trained the final model on all 30 features in Cell 33; tuned hyperparameters on `X_test`. | Performs **stratified train/test split (`stratify=y`) first**, fits all transformers strictly on `X_train`, and tunes hyperparameters + probability thresholds via 5-fold `GridSearchCV` / `cross_val_predict` on `X_train`. |
| **5. EDA: Raw Counts vs. Rates (%)** | Judged gender diversity and role turnover using raw headcount bars, skewing conclusions toward the 60% Male workforce baseline and high-headcount departments (`Research Scientist`). | Plots **normalized within-group percentages (%)** against company-wide baselines, revealing that `Sales Representative` has a **39.8% attrition rate** (nearly $2.5\times$ the company average). |
| **6. Feature Engineering & Collinearity** | Plotted an overlapping $31 \times 31$ correlation matrix and noted multicollinearity (`JobLevel` $\leftrightarrow$ `MonthlyIncome` $r = 0.95$), but engineered no features and dropped no collinear columns. | Masks the upper triangle (`fmt='.2f'`), drops redundant `JobLevel`, and engineers 4 HR domain features (`Overall_Satisfaction_Score`, `Income_Per_WorkingYear`, `Promotion_Stagnation_Ratio`, `Role_Tenure_Ratio`). |
| **7. Model Explainability** | Plotted unsorted Gini importance dots (`plt.plot(feature_importances_, 'o')`) with no indication of directionality. | Pairs **held-out Permutation Importance (PR-AUC impact)** with **standardized log-odds coefficients ($\beta$) and Odds Ratios ($e^\beta$)** to show both magnitude and direction. |

---

## Key Exploratory Data Analysis (EDA) Findings

1. **Class Imbalance ($5.2 : 1$)**:
   * Out of $1,470$ employees, **$1,233$ ($83.88\%$) stayed** and **$237$ ($16.12\%$) left**.
2. **Turnover by Job Role & Overtime**:
   * **Sales Representatives** experience a **39.8% attrition rate** ($33 / 83$), followed by **Laboratory Technicians (23.9%)** and **Human Resources (23.1%)**. Senior leadership roles (`Research Director` at **2.5%**, `Manager` at **4.9%**) have minimal turnover.
   * Employees working **OverTime (`Yes`)** leave at a **30.5% rate** ($127 / 416$) vs. only **10.4%** ($110 / 1,054$) for `OverTime = No` — nearly a **$3\times$ higher risk**.
3. **Business Travel Progression**:
   * Turnover scales monotonically with travel frequency: **Non-Travel (8.0%) < Travel_Rarely (15.0%) < Travel_Frequently (24.9%)**.
4. **Compensation, Tenure, & Stock Options**:
   * Employees who leave earn a **38.5% lower median monthly income** (**`$3,202`** vs. **`$5,204`** for stayers).
   * **65% of all leavers have zero stock options (`StockOptionLevel = 0`)**, and median company tenure for leavers is **3.0 years** (vs. **6.0 years** for stayers).
5. **Gender Diversity Re-evaluated**:
   * While the overall company is **60.0% Male / 40.0% Female**, normalizing within each role shows that **Manufacturing Director (49.7% Female)** and **Manager (46.1% Female)** exceed the company female baseline, and overall attrition rates between men (**17.0%**) and women (**14.8%**) are nearly identical.

---

## Modeling Results

### 1. 5-Fold Stratified Cross-Validation Benchmark ($N_{\text{train}} = 1,176$)

| Model | Train ROC-AUC | CV ROC-AUC (± SD) | CV PR-AUC | CV F1 | CV Recall | CV Precision | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced L2)** | 0.884 | **0.838 (±0.027)** | **0.651** | 0.503 | **0.747** | 0.382 | 0.761 |
| **Gradient Boosting (Regularized)** | 0.984 | 0.820 (±0.024) | 0.620 | 0.462 | 0.332 | **0.822** | **0.878** |
| **HistGradientBoosting (Balanced)** | 0.999 | 0.803 (±0.035) | 0.584 | **0.525** | 0.511 | 0.543 | 0.852 |
| **Random Forest (Regularized Balanced)** | 0.976 | 0.797 (±0.036) | 0.551 | 0.459 | 0.400 | 0.551 | 0.850 |
| **Decision Tree (Regularized)** | 0.831 | 0.712 (±0.011) | 0.358 | 0.441 | 0.605 | 0.349 | 0.751 |
| **Dummy Baseline (Most Frequent)** | 0.500 | 0.500 (±0.000) | 0.162 | 0.000 | 0.000 | 0.000 | 0.838 |

### 2. Held-Out Test Set Evaluation ($N_{\text{test}} = 294$, $47$ Positive Cases)

After hyperparameter tuning on `X_train` (`GridSearchCV` optimizing `average_precision`) and computing the out-of-fold F1-optimal probability threshold (`0.225`) for `GradientBoosting`:

| Model / Configuration | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 | Test Recall | Test Precision | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned Logistic Regression (Balanced L1, thr=0.50)** | 0.802 `[0.719, 0.875]` | **0.582 `[0.444, 0.712]`** | **0.478** | **0.681** | 0.368 | 0.762 |
| **Tuned Gradient Boosting (Default thr=0.50)** | **0.803 `[0.729, 0.868]`** | 0.486 `[0.351, 0.628]` | 0.328 | 0.213 | **0.714** | **0.861** |
| **Tuned Gradient Boosting (OOF-Tuned thr=0.22)** | **0.803 `[0.729, 0.868]`** | 0.486 `[0.351, 0.628]` | 0.472 | 0.532 | 0.424 | 0.810 |
| **Regularized Random Forest (Balanced, thr=0.50)** | 0.774 `[0.694, 0.846]` | 0.384 `[0.284, 0.523]` | 0.400 | 0.383 | 0.419 | 0.816 |

> **Key Takeaway**: At the default `0.50` threshold, `GradientBoosting` appears to have `86.1%` accuracy but misses **78.7% of employees who actually leave** (`Recall = 21.3%`). Tuning the decision threshold on out-of-fold predictions (`thr = 0.225`) more than doubles its Recall (`53.2%`), while **Tuned Balanced L1 Logistic Regression** catches **68.1% of leavers** (`32 / 47`) and achieves the highest **Test PR-AUC (`0.582`, $3.6\times$ the random baseline)**.

---

## Top Directional Drivers of Employee Attrition

From standardized L1 Logistic Regression log-odds ($\beta$) and Odds Ratios ($e^\beta$):

| Rank | Feature | Log-Odds ($\beta$) | Odds Ratio ($e^\beta$) | Direction & Business Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **`OverTime_Yes`** | `+1.148` | **`3.15x`** | Working overtime **more than triples** the odds of turnover. |
| 2 | **`Department_Research & Development`** | `-0.843` | **`0.43x`** | R&D staff have **57% lower odds** of leaving than Sales/HR. |
| 3 | **`JobRole_Laboratory Technician`** | `+0.673` | **`1.96x`** | Lab Technicians have nearly **2× higher odds** of leaving. |
| 4 | **`Overall_Satisfaction_Score`** *(Engineered)* | `-0.432` | **`0.65x`** | A $+1\text{ SD}$ increase in composite satisfaction cuts turnover odds by **35%**. |
| 5 | **`BusinessTravel`** | `+0.359` | **`1.43x`** | Higher travel frequency increases attrition odds by **43% per SD**. |
| 6 | **`NumCompaniesWorked`** | `+0.337` | **`1.40x`** | Prior job-hopping history increases turnover risk. |
| 7 | **`YearsSinceLastPromotion`** | `+0.306` | **`1.36x`** | Longer time since last promotion drives career stagnation turnover. |
| 8 | **`MonthlyIncome`** | `-0.285` | **`0.75x`** | Higher monthly income reduces turnover odds by **25% per SD**. |
| 9 | **`Role_Tenure_Ratio`** *(Engineered)* | `-0.271` | **`0.76x`** | Higher stability in current role relative to tenure reduces risk. |
| 10 | **`StockOptionLevel`** | `-0.256` | **`0.77x`** | Equity ownership (`StockOptionLevel >= 1`) significantly anchors retention. |

---

## Actionable HR Retention Recommendations

1. **Audit Workload & Overtime Allocation (`OverTime = Yes`)**:
   * Overtime is the #1 controllable predictor of attrition ($3.15\times$ odds ratio, $30.5\%$ attrition rate). Rebalancing workloads or hiring incremental headcount in high-overtime teams will directly pay for itself in avoided replacement costs.
2. **Restructure Early-Career Onboarding & Compensation for Sales Reps & Lab Techs**:
   * `Sales Representatives` ($39.8\%$ attrition) and `Laboratory Technicians` ($23.9\%$ attrition) in their first 3 years at the company (`YearsAtCompany <= 3`) represent the most acute flight-risk cohort. Review quota ramp-up timelines and entry-level base pay.
3. **Extend Broad-Based Equity Grants (`StockOptionLevel = 0 -> 1`)**:
   * $65\%$ of departing employees hold zero stock options. Introducing modest restricted stock / option grants for strong performers at the 2–3 year mark creates a powerful retention incentive.
4. **Trigger Proactive Stay-Interviews When Promotion Stagnation Combines with Travel**:
   * Use the calibrated risk score quarterly to flag employees with high `YearsSinceLastPromotion` and `Travel_Frequently` before they begin external job searches.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/AshutoshTamrakar1/HR-Analytics-Employee-Attrition.git
cd HR-Analytics-Employee-Attrition

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and open HR_Analytics_Case_Study_v2.ipynb
jupyter notebook HR_Analytics_Case_Study_v2.ipynb
```
