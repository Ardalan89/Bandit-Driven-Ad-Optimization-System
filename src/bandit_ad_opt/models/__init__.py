from .ucb import UCB1
from .thompson import ThompsonSampling

MODEL_REGISTRY = {
    "ucb": UCB1,
    "thompson_sampling": ThompsonSampling,
}

def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]