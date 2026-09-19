"""Run the complete immunotherapy-response model comparison and report."""
import argparse, json
from src.full_analysis import run_full_analysis
from src.immunotherapy_model import make_demo, prepare_data, read_table

def main():
    """Load demo or user data and run every analysis stage."""
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--demo", action="store_true")
    p.add_argument("--expression"); p.add_argument("--clinical"); p.add_argument("--sample-column", default="sample_id")
    p.add_argument("--target-column", default="response"); p.add_argument("--positive-label"); p.add_argument("--output-dir", default="results/full_analysis")
    p.add_argument("--test-size", type=float, default=.25); p.add_argument("--top-k", type=int, default=50); p.add_argument("--seed", type=int, default=42); a = p.parse_args()
    if a.demo: X, y, ids = make_demo(a.seed)
    else:
        if not a.expression or not a.clinical: p.error("provide both data files, or use --demo")
        X, y, ids = prepare_data(read_table(a.expression), read_table(a.clinical), a.sample_column, a.target_column, a.positive_label)
    print(json.dumps(run_full_analysis(X, y, ids, a.output_dir, a.test_size, a.top_k, a.seed, a.demo), indent=2))
if __name__ == "__main__": main()
