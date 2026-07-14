# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-07-14

### Added
- Part 1 - EDA: data loading, null analysis, duplicate detection, dtype correction, skewness
  analysis, IQR outlier detection, 6 required visualizations, mean/median imputation comparison,
  Spearman vs Pearson correlation analysis, grouped aggregation. Outputs cleaned_data.csv.
- Part 2 - Supervised ML: leak-free train/test split, one-hot encoding, Linear and Ridge
  regression, Logistic Regression classification with class-balance handling, ROC/AUC evaluation,
  decision-threshold sensitivity analysis, C-parameter regularization comparison, bootstrap
  confidence interval for AUC difference.
- Part 3 - Ensembles & Tuning: Decision Tree (unconstrained and constrained), Gini vs Entropy
  comparison, Random Forest, Gradient Boosting, feature-importance ablation study, 5-fold
  cross-validated model comparison, GridSearchCV hyperparameter tuning, manual learning curve,
  model serialization to best_model.pkl.
- Part 4 - LLM-Powered Feature: Track C (Model Prediction Explanation) - call_llm wrapper,
  PII guardrail, structured JSON-schema-validated explanations, temperature 0 vs 0.7 comparison.
- Root and per-Part README.md documentation, requirements.txt, .gitignore.
- LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md for open-source portfolio presentation.

### Notes
- Dataset (train.csv) arrived fully numeric with no missing values or duplicates; two
  categorical columns (Win_Tier, Run_Diff_Type) were engineered and disclosed, and synthetic
  missingness was injected into two columns to exercise the required data-cleaning tasks. See
  part1_eda/README.md - "Task 0" for full disclosure and justification.
