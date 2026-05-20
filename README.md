# Production Data Science Portfolio: Financial Risk Analytics and Human Capital Diagnostics

This repository contains a suite of end-to-end data science pipelines engineered to solve complex operational and risk management challenges in retail banking, consumer lending, and corporate human resource management. Each project is structured as an interview-grade production notebook utilizing rigorous data cleaning, advanced domain-specific feature engineering, statistical validation, and explainable machine learning frameworks.

---

## Repository Architecture and Core Notebooks

### 1. Bank Customer Churn Diagnostic Pipeline
* **File:** `Bank_Customer_Churn_Diagnostic_Pipeline.ipynb`
* **Domain:** Financial Risk Management / Customer Lifetime Value (LTV) Optimization
* **Core Objective:** Diagnose behavioral parameters driving voluntary attrition in retail banking and build a high-performance predictive model to flag detachment risks.
* **Methodology & Feature Engineering:**
  * Constructs 12+ interaction-driven risk markers including Tenure Asset Scaling Concentration (`BalancePerTenure`), Multi-Product Acquisition Velocity (`ProductsPerTenure`), Cash-Flow Liquidity Strain Indexes, and Multi-Product Footprint Engagement Scorecards.
  * Encodes categorical data paths using strict multi-collinearity safeguards (`drop_first=True`).
  * Establishes a multi-tiered meta-learning Stacking Classifier ensemble that combines a `GradientBoostingClassifier` with continuous `LogisticRegression` baselines to optimize prediction boundaries.

### 2. Consumer Loan Interest Rate Risk Assessment
* **File:** `Loan_Interest_Rate_Prediction.ipynb`
* **Domain:** Consumer Lending / Credit Risk Analytics
* **Core Objective:** Build an explainable parametric interest rate pricing model that evaluates a borrower's structural risk exposure.
* **Methodology & Feature Engineering:**
  * Implements granular data sanitization workflows, extracting quantitative credit indices from unstructured range strings to compute a reliable midpoint metric (`FICO_avg`).
  * Introduces corporate debt interaction metrics: Macro Debt Leverage (`loan_to_income`), Expected Amortization Cash Commitments (`monthly_payment`), Liquid Cash-Flow Strain (`payment_to_income`), and Resource Balance Stress Indexes (`credit_utilization`).
  * Deploys a standardized parametric Linear Regression matrix ensuring full feature coefficient accountability to satisfy strict regulatory model governance standards.

### 3. Corporate Human Capital Attrition Diagnostics
* **File:** `Attrition.ipynb`
* **Domain:** People Analytics / Human Capital Optimization
* **Core Objective:** Uncover behavioral and structural drivers of voluntary corporate turnover to assist talent retention efforts.
* **Methodology & Feature Engineering:**
  * Audits severe class imbalances (~83.1% retention vs. 16.9% attrition) to shift modeling benchmarks from raw accuracy to minority class precision, recall, and F1 optimization.
  * Derives psychological and tenure interaction parameters: Experience Normalization Indexes (`IncomePerYearExperience`), Career Track Promotion Gaps, Geographical Burnout Meters (`CommuteStress`), and Internal Pay Inequality Indicators (`IncomeVsDeptAvg`).
  * Trains an optimized `XGBClassifier` integrated with global feature importances and local SHAP (SHapley Additive exPlanations) beeswarm structures for explainable corporate decision-making.

---

## Analytical Methodology and Pipeline Standards

Every pipeline implemented within this portfolio adheres to identical engineering and documentation frameworks:

1. **Ingestion & Integrity Diagnostics:** Systematic automated evaluations of column signatures, missing value density matrix heatmaps, type-casting validations, and high-cardinality pruning.
2. **Behavioral Feature Synthesis:** Transforming static data points into multi-dimensional financial and behavioral ratios to build explicit linear separation boundaries for modeling layers.
3. **Statistical Exploration (EDA):** Validating structural parametric assumptions, target-stratified continuous distribution mapping, and global correlation analysis to systematically identify and neutralize multi-collinearity.
4. **Validation and Leakage Prevention:** Isolation of evaluation datasets using conservative cross-validation partitions prior to standard zero-mean unit-variance scaling layers (`StandardScaler`).
5. **Model Interpretability and Auditing:** Rejection of complete "black-box" conclusions; all model deployments are coupled with coordinate coefficient weights, ensemble feature importances, or SHAP value maps.

---

## Setup, Installation, and Execution

To replicate or review these execution pipelines locally, establish a virtual environment and verify that all necessary scientific computing packages are compiled.

### Prerequisites

Ensure you have a modern Python environment configured (Python >= 3.9 recommended).

```bash
# Clone the repository
git clone [https://github.com/Shaurya1126/portfolio-risk-analytics.git](https://github.com/Shaurya1126/portfolio-risk-analytics.git)
cd portfolio-risk-analytics

# Initialize a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the standardized dependency manifest
pip install -r requirements.txt
