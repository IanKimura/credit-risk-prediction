
## Model Performance

### Probability Calibration

The calibration plot compares predicted probabilities with observed repayment-difficulty rates across Logistic Regression, LightGBM, and XGBoost.

![Probability Calibration](images/calibration.png)

The final XGBoost model achieved a test Brier score of **0.0672**, with similar validation and test performance.

## Model Interpretability

SHAP (SHapley Additive exPlanations) was used to examine how individual features contribute to XGBoost's predictions.

### SHAP Summary

![SHAP Summary](images/shap_summary.png)

External credit-related scores, particularly `EXT_SOURCE_2` and `EXT_SOURCE_3`, were important predictors of repayment difficulties.

### SHAP Feature Importance

![SHAP Feature Importance](images/shap_importance.png)

The SHAP importance plot summarizes the average magnitude of each feature's contribution to model predictions.

## Cost-Sensitive Threshold Optimization

Rather than applying the default classification threshold of 0.50, the decision threshold was optimized using validation data and hypothetical misclassification costs:

- False positive: **$500**
- False negative: **$5,000**

![Threshold Optimization](images/threshold_optimization.png)

The selected threshold of **0.086** minimized the observed hypothetical classification cost on the validation set within the evaluated thresholds.

## Business Impact

The final decision strategy was evaluated on the untouched test dataset.

![Business Impact](images/business_impact.png)

Using the frozen 8.6% threshold, XGBoost achieved a **36.16% reduction in hypothetical classification costs** relative to flagging no borrowers.

These results demonstrate how machine learning predictions can be translated into cost-sensitive financial decisions. The costs are illustrative and do not represent realized lending savings.
