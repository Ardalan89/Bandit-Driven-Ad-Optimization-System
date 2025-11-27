from bandit_ad_opt.models.thompson import ThompsonSampling

def test_thompson_basic():
    model = ThompsonSampling(n_arms=2, alpha=1.0, beta=1.0)

    # Ensure select_arm returns a valid arm
    arm = model.select_arm()
    assert arm in [0, 1]

    # Update with reward
    model.update(arm, 1)
