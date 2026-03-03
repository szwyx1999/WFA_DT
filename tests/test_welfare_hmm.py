import numpy as np
from src.models.welfare_hmm.hmm import GaussianHMM


def test_hmm_learns_two_regimes_and_risk_increases():
    # Make a toy 1D sequence with a clear regime shift
    rng = np.random.default_rng(0)
    T = 80
    X = np.zeros((T, 1), dtype=float)
    X[:30, 0] = rng.normal(1.0, 0.2, size=30)      # "normal": higher mean
    X[30:60, 0] = rng.normal(-1.0, 0.2, size=30)   # "stress": lower mean
    X[60:, 0] = rng.normal(0.8, 0.2, size=20)      # back closer to normal

    hmm = GaussianHMM(K=2, D=1, random_state=0, var_floor=1e-3)
    ll_hist = hmm.fit([X], weights=[np.ones(T)], n_iter=15, tol=1e-5)

    # reorder: state0 should be the higher-mean one (normal)
    hmm.reorder_states_by_feature(feature_index=0, descending=True)

    gamma = hmm.predict_proba(X)
    risk = 1.0 - gamma[:, 0]  # P(not normal)

    assert risk[30:60].mean() > risk[:30].mean()
    assert risk[30:60].mean() > 0.6