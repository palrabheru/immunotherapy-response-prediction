"""Leakage-safe binary classification for gene-expression data."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POSITIVE_LABELS = {"1", "true", "yes", "responder", "response", "cr", "pr"}

class SafeSelectKBest(SelectKBest):
    """Cap k after upstream variance filtering."""
    def fit(self, X, y):
        self.k = min(self.k, X.shape[1])
        return super().fit(X, y)

def read_table(path):
    """Read CSV/TSV input, including gzip-compressed files."""
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep="\t" if ".tsv" in path.name.lower() or ".txt" in path.name.lower() else ",")

def encode_target(series, positive_label=None):
    """Convert a two-class clinical endpoint to 0 (negative) and 1 (positive)."""
    if series.isna().any(): raise ValueError("Target column contains missing values.")
    values = series.astype(str).str.strip().str.lower(); unique = set(values.unique())
    if len(unique) != 2: raise ValueError(f"Target must have exactly two classes; found {sorted(unique)}")
    matches = unique & POSITIVE_LABELS
    positive = positive_label.strip().lower() if positive_label else (matches.pop() if len(matches) == 1 else None)
    if positive not in unique: raise ValueError("Cannot identify positive class; pass --positive-label.")
    return (values == positive).astype(int)

def prepare_data(expression, clinical, sample_column, target_column, positive_label=None):
    """Join input tables by sample ID and return validated model-ready values."""
    for frame, col, label in ((expression, sample_column, "expression"), (clinical, sample_column, "clinical"), (clinical, target_column, "clinical")):
        if col not in frame: raise ValueError(f"Column {col!r} is missing from {label} data.")
    if expression[sample_column].duplicated().any() or clinical[sample_column].duplicated().any(): raise ValueError("Sample IDs must be unique.")
    merged = expression.merge(clinical[[sample_column, target_column]], on=sample_column)
    if len(merged) < 20: raise ValueError(f"At least 20 matched samples are required; found {len(merged)}.")
    y = encode_target(merged[target_column], positive_label)
    if y.value_counts().min() < 5: raise ValueError("Each class needs at least five samples.")
    X = merged.drop(columns=[sample_column, target_column]).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if X.shape[1] < 2: raise ValueError("At least two numeric gene features are required.")
    return X, y, merged[sample_column].astype(str)

def make_demo(seed=42, n_samples=160, n_genes=120):
    """Create deterministic synthetic data for installation checks and tutorials."""
    informative = min(14, max(2, n_genes // 2)); redundant = min(8, max(0, n_genes - informative - 1))
    X, y = make_classification(n_samples=n_samples, n_features=n_genes, n_informative=informative, n_redundant=redundant, weights=[.65, .35], class_sep=1.1, random_state=seed)
    return pd.DataFrame(X, columns=[f"GENE_{i:03d}" for i in range(n_genes)]), pd.Series(y), pd.Series([f"DEMO_{i:03d}" for i in range(n_samples)])

def train_and_save(X, y, ids, output_dir, test_size=.25, k=50, seed=42):
    """Fit on a stratified training split and save held-out evaluation artifacts."""
    if not .1 <= test_size <= .5: raise ValueError("test_size must be between 0.1 and 0.5.")
    train, test = train_test_split(np.arange(len(X)), test_size=test_size, stratify=y, random_state=seed)
    # Every data-dependent transformation stays inside the pipeline. This ensures
    # that test-set information cannot leak into imputation or feature selection.
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("variance", VarianceThreshold()),
        ("select", SafeSelectKBest(f_classif, k=min(k, X.shape[1]))), ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))])
    model.fit(X.iloc[train], y.iloc[train]); probability = model.predict_proba(X.iloc[test])[:, 1]
    prediction = (probability >= .5).astype(int); truth = y.iloc[test].to_numpy()
    metrics = {"n_samples": len(X), "n_genes": X.shape[1], "n_test": len(test), "positive_rate": y.mean(),
        "accuracy": accuracy_score(truth, prediction), "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "precision": precision_score(truth, prediction, zero_division=0), "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0), "roc_auc": roc_auc_score(truth, probability),
        "average_precision": average_precision_score(truth, probability), "seed": seed}
    metrics = {key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame({"sample_id": ids.iloc[test].to_numpy(), "true_label": truth, "predicted_label": prediction, "response_probability": probability}).to_csv(out / "predictions.csv", index=False)
    # Apply both feature masks in pipeline order so coefficients map back to the
    # correct original gene names.
    names = X.columns[model["variance"].get_support()]; names = names[model["select"].get_support()]; coefs = model["model"].coef_[0]
    pd.DataFrame({"gene": names, "coefficient": coefs, "absolute_coefficient": np.abs(coefs)}).sort_values("absolute_coefficient", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    ConfusionMatrixDisplay(confusion_matrix(truth, prediction), display_labels=["non-response", "response"]).plot(cmap="Blues", colorbar=False)
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=160); plt.close(); return metrics
