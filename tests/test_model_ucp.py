import numpy as np
from bandit_ad_opt.models.ucb import UCB1

def test_ucb_basic():
    model = UCB1(n_arms=3, confidence=2.0)

    # Force first round: all counts=0, so arm 0 must be chosen
    arm = model.select_arm()
    assert arm == 0

    # Update arm 0 with reward
    model.update(0, 1)

    # Now arm 1 and 2 have count 0 → arm 1 should be chosen next
    arm = model.select_arm()
    assert arm == 1
