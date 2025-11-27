import numpy as np

class ThompsonSampling:
    """
    Thompson Sampling for Bernoulli multi-armed bandits.
    Uses Beta posterior for reward distribution.
    """

    def __init__(self, n_arms: int, alpha: float = 1, beta: float = 1):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha
        self.beta = np.ones(n_arms) * beta

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """
        reward = 1 or 0 (Bernoulli)
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
