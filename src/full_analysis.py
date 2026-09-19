"""Comprehensive model comparison, visualization, and reporting workflow."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .immunotherapy_model import SafeSelectKBest


SCORING = {
    "roc_auc": "roc_auc", "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy", "f1": "f1",
}


def _pipeline(estimator, n_features: int, top_k: int) -> Pipeline:
    """Build an identical leakage-safe preprocessing path for every model."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("select", SafeSelectKBest(f_classif, k=min(top_k, n_features))),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def candidate_models(n_features: int, top_k: int, seed: int) -> dict[str, Pipeline]:
    """Return interpretable linear, nonlinear tree, and kernel baselines."""
    return {
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed), n_features, top_k),
        "random_forest": _pipeline(
            RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced",
                                   random_state=seed, n_jobs=-1), n_features, top_k),
        "support_vector_machine": _pipeline(
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed), n_features, top_k),
    }


def _save_eda(X: pd.DataFrame, y: pd.Series, out: Path) -> None:
    """Save class balance, library-size proxy, and two-dimensional PCA figures."""
    figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4)); plt.bar(["Non-response", "Response"], counts.reindex([0, 1], fill_value=0), color=["#64748b", "#0f766e"])
    plt.ylabel("Samples"); plt.title("Response class distribution"); plt.tight_layout(); plt.savefig(figures / "class_distribution.png", dpi=180); plt.close()

    numeric = SimpleImputer(strategy="median").fit_transform(X)
    transformed = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=2, random_state=0).fit_transform(transformed)
    plt.figure(figsize=(6, 5))
    for label, name, color in [(0, "Non-response", "#64748b"), (1, "Response", "#0f766e")]:
        mask = y.to_numpy() == label; plt.scatter(pca[mask, 0], pca[mask, 1], s=28, alpha=.75, label=name, c=color)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of expression profiles"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "pca_by_response.png", dpi=180); plt.close()

    sample_medians = X.median(axis=1)
    plt.figure(figsize=(6, 4)); plt.hist(sample_medians, bins=25, color="#2563eb", edgecolor="white")
    plt.xlabel("Median expression across genes"); plt.ylabel("Samples"); plt.title("Sample-level expression distribution")
    plt.tight_layout(); plt.savefig(figures / "sample_expression_distribution.png", dpi=180); plt.close()


def _evaluate(y_true, probability) -> dict[str, float | list[list[int]]]:
    """Compute threshold and ranking metrics from positive-class probabilities."""
    predicted = (probability >= .5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
    }


def run_full_analysis(X: pd.DataFrame, y: pd.Series, ids: pd.Series, output_dir,
                      test_size=.25, top_k=50, seed=42, demo=False) -> dict:
    """Compare models by cross-validation, evaluate them, and save a full report."""
    out = Path(output_dir); (out / "tables").mkdir(parents=True, exist_ok=True)
    _save_eda(X, y, out)
    train, test = train_test_split(np.arange(len(X)), test_size=test_size, stratify=y, random_state=seed)
    X_train, X_test = X.iloc[train], X.iloc[test]; y_train, y_test = y.iloc[train], y.iloc[test]
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2: raise ValueError("Not enough training samples per class for cross-validation.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = candidate_models(X.shape[1], top_k, seed)
    cv_rows, test_results, probabilities = [], {}, {}

    for name, model in models.items():
        # Model choice uses training-set cross-validation only; the test set is
        # evaluated afterward and never influences selection.
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[f"mean_{metric}"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"std_{metric}"] = float(np.std(scores[f"test_{metric}"]))
        cv_rows.append(row)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        probabilities[name] = probability; test_results[name] = _evaluate(y_test, probability)

    cv_table = pd.DataFrame(cv_rows).sort_values("mean_roc_auc", ascending=False)
    cv_table.to_csv(out / "tables" / "cross_validation_results.csv", index=False)
    best_name = str(cv_table.iloc[0]["model"]); best_model = models[best_name]
    joblib.dump(best_model, out / "best_model.joblib")

    predictions = pd.DataFrame({"sample_id": ids.iloc[test].to_numpy(), "true_label": y_test.to_numpy()})
    for name, probability in probabilities.items(): predictions[f"{name}_probability"] = probability
    predictions.to_csv(out / "tables" / "test_predictions.csv", index=False)

    importance = permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=-1)
    importance_table = pd.DataFrame({"gene": X.columns, "importance_mean": importance.importances_mean,
                                     "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(out / "tables" / "permutation_importance.csv", index=False)

    figures = out / "figures"
    plt.figure(figsize=(7, 4)); plt.barh(cv_table["model"], cv_table["mean_roc_auc"], xerr=cv_table["std_roc_auc"], color="#0f766e")
    plt.xlim(0, 1); plt.xlabel("Mean cross-validated ROC AUC"); plt.title("Model comparison"); plt.tight_layout(); plt.savefig(figures / "model_comparison.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability); plt.plot(fpr, tpr, label=f"{name} ({test_results[name]['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate"); plt.title("Held-out ROC curves"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figures / "roc_curves.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability); plt.plot(recall, precision, label=f"{name} ({test_results[name]['average_precision']:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Held-out precision-recall curves"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figures / "precision_recall_curves.png", dpi=180); plt.close()
    top = importance_table.head(20).sort_values("importance_mean")
    plt.figure(figsize=(7, 6)); plt.barh(top["gene"], top["importance_mean"], xerr=top["importance_std"], color="#2563eb")
    plt.xlabel("Decrease in ROC AUC after permutation"); plt.title(f"Top features: {best_name}"); plt.tight_layout(); plt.savefig(figures / "top_gene_importance.png", dpi=180); plt.close()

    summary = {"dataset": "synthetic demonstration" if demo else "user-supplied cohort", "n_samples": len(X),
               "n_genes": X.shape[1], "test_samples": len(test), "cv_folds": folds,
               "selected_model": best_name, "cross_validation": cv_rows, "held_out_test": test_results, "seed": seed}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    disclaimer = "These values come from synthetic data and have no clinical meaning." if demo else "Results require independent external validation before scientific or clinical interpretation."
    report = f"""# Immunotherapy Response Analysis Report

## Scope

{disclaimer}

The workflow analyzed **{len(X)} samples** and **{X.shape[1]} expression features**. A stratified split reserved **{len(test)} samples** for final evaluation. Model selection used {folds}-fold cross-validation on the training set only.

## Model selection

The selected model was **{best_name.replace('_', ' ')}**, based on mean training-set cross-validated ROC AUC. Complete cross-validation values are in `tables/cross_validation_results.csv`; held-out predictions are in `tables/test_predictions.csv`.

## Interpretation

Permutation importance ranks features by the decrease in held-out ROC AUC when each gene is shuffled. This is predictive evidence, not proof of biological causation. Correlated genes can divide importance between them.

## Limitations

- Small cohorts can produce unstable estimates and wide uncertainty.
- Batch, tumor-type, treatment, and patient-level effects can confound results.
- Hyperparameter tuning and external validation are intentionally not represented as completed clinical validation.
- Response definitions must be harmonized before combining cohorts.
"""
    (out / "REPORT.md").write_text(report)
    return summary
