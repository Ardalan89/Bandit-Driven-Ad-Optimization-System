import numpy as np
from bandit_ad_opt.data_loader import load_ctr_data
from bandit_ad_opt.models import get_model_class

def run_bandit_experiment(cfg, model_name):
    """ Run bandit experiment with given model and config."""

    # Load data and set number of arms
    df, n_arms = load_ctr_data(cfg["data"]["path"])

    # Resolve model class
    ModelClass = get_model_class(model_name)  


    # Filter parameters for the model
    params = cfg["bandit"]["params"]
    valid = ModelClass.__init__.__code__.co_varnames
    filtered_params = {k: v for k, v in params.items() if k in valid}

    # Instantiate model with correct n_arms
    model = ModelClass(n_arms=n_arms, **filtered_params)

    rounds = cfg["bandit"]["rounds"]
    rewards = []
    chosen_arms = []

    for i in range(rounds):
        arm = model.select_arm()
        chosen_arms.append(arm)

        reward = int(df.iloc[i % len(df), arm])
        rewards.append(reward)

        model.update(arm, reward)

    return {
        "algorithm": model_name,
        "rewards": np.array(rewards),
        "chosen_arms": np.array(chosen_arms),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
    }
