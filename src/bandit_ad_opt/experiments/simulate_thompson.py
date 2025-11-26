from src.bandit_ad_opt.bandits.thompson import ThompsonSampling
from src.bandit_ad_opt.data_loader import load_ctr_data
import numpy as np

def run_thompson(global_config, algo_config):
    df, n_arms = load_ctr_data(
        path=global_config["data"]["path"],
        ad_column=global_config["data"]["ad_column"],
        click_column=global_config["data"]["click_column"]
    )

    ts = ThompsonSampling(
        n_arms=n_arms,
        alpha=algo_config["thompson"]["alpha"],
        beta=algo_config["thompson"]["beta"],
    )

    rounds = algo_config["rounds"]
    rewards = []
    chosen_arms = []

    for i in range(rounds):
        arm = ts.select_arm()
        chosen_arms.append(arm)

        reward = df.iloc[i % len(df)]["reward"]
        rewards.append(reward)

        ts.update(arm, reward)

    return {
        "algorithm": "thompson",
        "rewards": np.array(rewards),
        "chosen_arms": np.array(chosen_arms),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
    }
