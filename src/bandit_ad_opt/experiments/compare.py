from src.bandit_ad_opt.experiments.simulate_ucb import run_ucb
from src.bandit_ad_opt.experiments.simulate_thompson import run_thompson

def compare(global_config, algo_config):
    algo_type = algo_config["type"]

    if algo_type == "ucb":
        return run_ucb(global_config, algo_config)
    elif algo_type == "thompson":
        return run_thompson(global_config, algo_config)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
