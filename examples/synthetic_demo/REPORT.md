# Immunotherapy Response Analysis Report

## Scope

These values come from synthetic data and have no clinical meaning.

The workflow analyzed **160 samples** and **120 expression features**. A stratified split reserved **40 samples** for final evaluation. Model selection used 5-fold cross-validation on the training set only.

## Model selection

The selected model was **support vector machine**, based on mean training-set cross-validated ROC AUC. Complete cross-validation values are in `tables/cross_validation_results.csv`; held-out predictions are in `tables/test_predictions.csv`.

## Interpretation

Permutation importance ranks features by the decrease in held-out ROC AUC when each gene is shuffled. This is predictive evidence, not proof of biological causation. Correlated genes can divide importance between them.

## Limitations

- Small cohorts can produce unstable estimates and wide uncertainty.
- Batch, tumor-type, treatment, and patient-level effects can confound results.
- Hyperparameter tuning and external validation are intentionally not represented as completed clinical validation.
- Response definitions must be harmonized before combining cohorts.
