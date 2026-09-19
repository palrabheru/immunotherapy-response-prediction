from src.full_analysis import run_full_analysis
from src.immunotherapy_model import make_demo

def test_full_analysis_outputs(tmp_path):
    X, y, ids = make_demo(n_samples=80, n_genes=24)
    summary = run_full_analysis(X, y, ids, tmp_path, top_k=10, demo=True)
    assert summary["selected_model"] in {"logistic_regression", "random_forest", "support_vector_machine"}
    expected = ["analysis_summary.json", "REPORT.md", "best_model.joblib", "tables/cross_validation_results.csv",
                "tables/test_predictions.csv", "tables/permutation_importance.csv", "figures/roc_curves.png",
                "figures/precision_recall_curves.png", "figures/pca_by_response.png", "figures/top_gene_importance.png"]
    assert all((tmp_path / path).exists() for path in expected)
