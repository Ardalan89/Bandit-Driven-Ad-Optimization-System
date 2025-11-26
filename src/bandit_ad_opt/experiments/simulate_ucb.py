from src.bandit_ad_opt.bandits.ucb import UCB1
from src.bandit_ad_opt.data_loader import load_ctr_data
import numpy as np

def run_ucb(global_config, algo_config):
    """
    Run UCB1 on real CTR dataset.
    
    global_config: main experiment settings (default.yaml)
    algo_config: specific algorithm config (configs/ucb.yaml)
    """
    df, n_arms = load_ctr_data(
        path=global_config["data"]["path"],
        ad_column=global_config["data"]["ad_column"],
        click_column=global_config["data"]["click_column"]
    )

    ucb = UCB1(
        n_arms=n_arms,
        confidence=algo_config["ucb"]["confidence"]
    )

    rounds = algo_config["rounds"]
    rewards = []
    chosen_arms = []

    # iterate through dataset (or loop if dataset < rounds)
    for i in range(rounds):
        arm = ucb.select_arm()
        chosen_arms.append(arm)

        # get reward from dataset row i (cyclic)
        reward = df.iloc[i % len(df)]["reward"]
        rewards.append(reward)

        ucb.update(arm, reward)

    return {
        "algorithm": "ucb",
        "rewards": np.array(rewards),
        "chosen_arms": np.array(chosen_arms),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
    }
