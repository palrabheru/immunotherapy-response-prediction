import pandas as pd
from src.immunotherapy_model import encode_target, make_demo, train_and_save
def test_target_encoding(): assert encode_target(pd.Series(["Responder", "Non-responder"])).tolist() == [1, 0]
def test_demo_pipeline(tmp_path):
    X, y, ids = make_demo(n_samples=80, n_genes=30); metrics = train_and_save(X, y, ids, tmp_path, k=10)
    assert 0 <= metrics["roc_auc"] <= 1
    assert all((tmp_path / n).exists() for n in ["model.joblib", "metrics.json", "predictions.csv", "feature_importance.csv", "confusion_matrix.png"])
