
# Credit Risk Prediction & Cost-Sensitive Decision Optimization

### End-to-End Machine Learning | Financial Risk Analytics | Explainable AI

## Project Overview

This project develops an end-to-end machine learning pipeline to predict borrower repayment difficulties using the **Home Credit Default Risk** dataset.

The objective is not only to predict credit risk accurately, but also to explore how predicted probabilities can support financial decision-making when different classification errors have different costs.

The project compares Logistic Regression, Random Forest, LightGBM, and XGBoost, evaluates model calibration and interpretability, and optimizes a decision threshold under hypothetical financial costs.

## Key Results

| Metric | Final Test Result |
|---|---:|
| ROC-AUC | 0.7691 |
| PR-AUC | 0.2546 |
| Brier Score | 0.0672 |
| Precision | 17.82% |
| Recall | 67.11% |
| F1-Score | 0.2816 |
| Selected Threshold | 8.6% |

**Business impact:** The selected XGBoost decision rule achieved a **36.16% reduction in hypothetical classification costs** compared with flagging no borrowers.

These savings are simulated under fixed cost assumptions and do not represent realized lending profits.

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

The primary application dataset contains 307,511 borrower observations, with approximately 8.07% labeled as experiencing repayment difficulties.

The data was divided into stratified training (70%), validation (15%), and test (15%) sets.

## Methodology

### 1. Exploratory Data Analysis

Investigated class imbalance, missing values, borrower characteristics, financial variables, and relationships with repayment difficulties.

### 2. Feature Engineering and Preprocessing

Created financial ratios and borrower characteristics, including age, employment duration, credit-to-income ratio, and annuity-to-income ratio.

Applied missing-value imputation, numerical scaling, and categorical encoding using training-data-fitted transformations.

### 3. Model Development

Compared four machine learning approaches:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

Model selection was based on validation performance, including ROC-AUC, PR-AUC, and probability calibration.

### 4. Model Explainability

Used SHAP values to examine feature contributions and explain individual borrower predictions.

External credit-related scores emerged as important predictive features.

### 5. Cost-Sensitive Threshold Optimization

Evaluated decision thresholds using hypothetical costs of:

- False positive: $500
- False negative: $5,000

Selected an 8.6% decision threshold using validation data.

### 6. Independent Test Evaluation

Evaluated the final model and frozen decision threshold on 46,127 previously unseen observations.

No further model tuning or threshold optimization was performed using the test set.

## Business Impact

| Decision Strategy | Hypothetical Test Cost |
|---|---:|
| Flag Nobody | $18,620,000 |
| Flag Everybody | $21,201,500 |
| XGBoost (8.6%) | $11,886,500 |

The XGBoost decision strategy reduced simulated classification costs by 36.16% relative to flagging nobody.

This illustrates how probability predictions and business-specific cost assumptions can be combined to support risk-based decisions.

## Repository Structure

```text
notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_modeling.ipynb
    04_evaluation.ipynb
    05_final_results.ipynb
```

## Tools and Technologies

Python, Pandas, NumPy, scikit-learn, LightGBM, XGBoost, SHAP, Matplotlib, SciPy, and Google Colab.

## Limitations and Future Work

This project uses a historical dataset and a random train-validation-test split. It does not establish performance under future economic conditions or changing borrower populations.

The financial cost assumptions are illustrative rather than institution-specific. Real-world deployment would require additional calibration validation, fairness and compliance assessments, temporal testing, and monitoring.

Future extensions could incorporate supplementary Home Credit datasets, more realistic loss modeling, and out-of-time validation.

## Project Notebooks

The five notebooks document the workflow from exploratory analysis through final model evaluation.

The raw dataset is available from Kaggle. The notebooks were developed in Google Colab and use project-specific Google Drive paths; additional setup is required to reproduce the workflow in another environment.
