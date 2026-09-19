# Immunotherapy Response Prediction

Reproducible machine-learning baseline for predicting binary immunotherapy response from gene-expression data. It validates inputs, prevents preprocessing leakage, handles class imbalance, evaluates held-out samples, ranks genes, and saves a reusable model.

> Research and education only. This is not a medical device and must not guide clinical decisions.

## Quick start

```bash
git clone https://github.com/palrabheru/immunotherapy-response-prediction.git
cd immunotherapy-response-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --demo --output-dir outputs/demo
pytest -q
```

## Real data

Expression must contain one row per sample, a unique ID column, and numeric gene columns. Clinical data must contain the matching ID and a binary response column. CSV, TSV, and gzip-compressed versions are supported.

```bash
python train.py --expression data/processed/expression.csv \
  --clinical data/processed/clinical.csv --sample-column sample_id \
  --target-column response --positive-label Responder \
  --output-dir outputs/real_run
```

If genes are rows, transpose the matrix first. Default positive labels include `1`, `true`, `yes`, `responder`, `response`, `cr`, and `pr`; otherwise pass `--positive-label`.

## Method and outputs

The code uses a stratified train/test split and fits median imputation, variance filtering, univariate feature selection, standardization, and class-balanced logistic regression on training data only.

| Output | Description |
|---|---|
| `model.joblib` | Complete fitted pipeline |
| `metrics.json` | Accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and average precision |
| `predictions.csv` | Held-out labels and probabilities |
| `feature_importance.csv` | Selected genes ranked by coefficient magnitude |
| `confusion_matrix.png` | Held-out confusion matrix |

```text
train.py                    Command-line entry point
src/immunotherapy_model.py  Validation, modeling, and output logic
tests/                      Automated tests
notebooks/                  Exploratory work retained for reference
data/                       Local data directories (large data not committed)
```

The seed defaults to `42`. Coefficients are predictive associations, not causal biomarkers. Before making scientific claims, add external-cohort validation, patient-level grouped splitting where needed, leakage-safe batch correction, calibration, uncertainty estimates, and biological validation. The original notebook references **GSE176307**; verify that the chosen metadata actually provides the required treatment-response endpoint.

## License

MIT. See `LICENSE`.
