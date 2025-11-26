import numpy as np


class UCB1:
    """
    UCB1 implementation for multi-armed bandits.
    Tracks number of pulls and estimated values for each arm.
    """
    def __init__(self, n_arms: int, confidence: float = 2.0):
        self.n_arms = n_arms
        self.confidence = confidence
        self.counts = np.zeros(n_arms)  
        self.values = np.zeros(n_arms) 
        self.total_pulls = 0  
    
    def select_arm(self) -> int:
        """
        Select an arm using the UCB1 strategy
        """
        
        # if any arm has not been pulled yet, select it
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # calculate UBC scores
        exploration = np.sqrt((self.confidence * np.log(self.total_pulls)) / self.counts)
        ucb_values = self.values + exploration  
        return int(np.argmax(ucb_values))
    
    def update(self, arm: int, reward: float):
        """
        Update running estimates after observing reward.
        reward must be numeric (0/1 for Bernoulli reward).
        """
        self.total_pulls += 1
        self.counts[arm] += 1
        
        # incremental mean update
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        
               
        