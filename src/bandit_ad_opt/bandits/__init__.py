from .ucb import UCB1
from .thompson import ThompsonSampling

ALGO_REGISTRY = {
    "ucb": UCB1,
    "thompson": ThompsonSampling,
}