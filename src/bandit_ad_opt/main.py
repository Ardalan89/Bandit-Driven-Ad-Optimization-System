import argparse
from pathlib import Path
import numpy as np

from bandit_ad_opt.utils.config import load_config
from bandit_ad_opt.experiments.runner import run_bandit_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Override model: ucb, thompson_sampling")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config("configs/default.yaml")

    model_name = args.model if args.model else cfg["model"]

    results = run_bandit_experiment(cfg, model_name)

    print("========== RESULTS ==========")
    print("Model:", results["algorithm"])
    print("Total Reward:", results["total_reward"])
    print("Mean Reward:", results["mean_reward"])

    # Save results
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{results['algorithm']}_results.npy", results)

if __name__ == "__main__":
    main()
