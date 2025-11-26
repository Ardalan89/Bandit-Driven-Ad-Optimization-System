import numpy as np

def sample_bernoulli_reward(true_ctr: float) -> int:
    """
    Sample a binary reward (0/1) using the given CTR probability.
    Used for synthetic simulations.
    """
    return int(np.random.rand() < true_ctr)
