from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


def logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """Tiny logsumexp, no scipy needed."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def gaussian_logpdf_diag(X: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    """
    Diagonal Gaussian emission log p(x_t | z=k).
    X: (T, D), means: (K, D), vars_: (K, D)
    returns: (T, K)
    """
    T, D = X.shape
    K = means.shape[0]
    # (T, K, D)
    diff2 = (X[:, None, :] - means[None, :, :]) ** 2
    log_det = np.sum(np.log(2.0 * np.pi * vars_), axis=1)  # (K,)
    quad = np.sum(diff2 / vars_[None, :, :], axis=2)       # (T, K)
    return -0.5 * (log_det[None, :] + quad)


def normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return A / s


@dataclass
class GaussianHMM:
    """
    A minimal Gaussian HMM (diag cov) with EM training.
    Not fancy, but good enough for a 1-day demo + math story.
    """
    K: int
    D: int
    var_floor: float = 1e-3
    eps: float = 1e-12
    random_state: int = 42

    pi: np.ndarray = None      # (K,)
    A: np.ndarray = None       # (K, K)
    means: np.ndarray = None   # (K, D)
    vars_: np.ndarray = None   # (K, D)

    def _init_params(self, X_all: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)

        # init pi close to normal state
        pi = np.ones(self.K) / self.K
        pi[0] = 0.8
        pi = pi / pi.sum()

        # init A with strong self-transition (sticky states)
        A = np.full((self.K, self.K), 1.0 / self.K)
        for k in range(self.K):
            A[k, :] *= 0.2
            A[k, k] = 0.8
        A = normalize_rows(A, self.eps)

        # init emission params by random sampling points
        idx = rng.choice(len(X_all), size=self.K, replace=False) if len(X_all) >= self.K else np.arange(self.K) % len(X_all)
        means = X_all[idx].copy()
        vars_ = np.var(X_all, axis=0, keepdims=True).repeat(self.K, axis=0)
        vars_ = np.maximum(vars_, self.var_floor)

        self.pi, self.A, self.means, self.vars_ = pi, A, means, vars_

    def _forward_backward(self, X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        returns:
          loglik, gamma(T,K), xi(T-1,K,K), logB(T,K)
        """
        T = X.shape[0]
        logB = gaussian_logpdf_diag(X, self.means, self.vars_)  # (T,K)
        logA = np.log(np.maximum(self.A, self.eps))
        logpi = np.log(np.maximum(self.pi, self.eps))

        # forward
        alpha = np.zeros((T, self.K), dtype=float)
        alpha[0] = logpi + logB[0]
        for t in range(1, T):
            # alpha[t,j] = logB[t,j] + logsum_i alpha[t-1,i] + logA[i,j]
            alpha[t] = logB[t] + logsumexp(alpha[t - 1][:, None] + logA, axis=0)

        loglik = float(logsumexp(alpha[T - 1], axis=0))

        # backward
        beta = np.zeros((T, self.K), dtype=float)
        beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            # beta[t,i] = logsum_j logA[i,j] + logB[t+1,j] + beta[t+1,j]
            beta[t] = logsumexp(logA + (logB[t + 1] + beta[t + 1])[None, :], axis=1)

        # gamma
        log_gamma = alpha + beta - loglik
        gamma = np.exp(log_gamma)
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), self.eps)

        # xi for transitions
        xi = np.zeros((max(0, T - 1), self.K, self.K), dtype=float)
        if T > 1:
            for t in range(T - 1):
                log_xi_t = (
                    alpha[t][:, None]
                    + logA
                    + (logB[t + 1] + beta[t + 1])[None, :]
                    - loglik
                )
                xi_t = np.exp(log_xi_t)
                s = np.maximum(xi_t.sum(), self.eps)
                xi[t] = xi_t / s

        return loglik, gamma, xi, logB

    def fit(self, sequences: List[np.ndarray], weights: Optional[List[np.ndarray]] = None,
            n_iter: int = 10, tol: float = 1e-4) -> List[float]:
        """
        EM training. weights[t] is in [0,1] and downweights bad rows (e.g. anomalies).
        """
        if weights is None:
            weights = [np.ones(seq.shape[0], dtype=float) for seq in sequences]

        X_all = np.concatenate(sequences, axis=0)
        if self.pi is None:
            self._init_params(X_all)

        ll_hist: List[float] = []
        last_ll = None

        for it in range(n_iter):
            # accumulators
            pi_acc = np.zeros(self.K)
            A_num = np.zeros((self.K, self.K))
            A_den = np.zeros(self.K)

            w_sum = np.zeros(self.K)
            x_sum = np.zeros((self.K, self.D))
            x2_sum = np.zeros((self.K, self.D))

            total_ll = 0.0

            for X, w in zip(sequences, weights):
                T = X.shape[0]
                ll, gamma, xi, _ = self._forward_backward(X)
                total_ll += ll

                # weight vector (T,)
                w = w.astype(float)
                w = np.clip(w, 0.0, 1.0)

                # pi update (use first step weight)
                pi_acc += gamma[0] * (w[0] if T > 0 else 1.0)

                # transitions
                if T > 1:
                    # transition weights: if either endpoint is "bad", we downweight
                    w_trans = np.minimum(w[:-1], w[1:])  # (T-1,)
                    for t in range(T - 1):
                        A_num += xi[t] * w_trans[t]
                        A_den += gamma[t] * w_trans[t]

                # emissions (weighted)
                for k in range(self.K):
                    wk = gamma[:, k] * w
                    sw = wk.sum()
                    w_sum[k] += sw
                    if sw > 0:
                        x_sum[k] += (wk[:, None] * X).sum(axis=0)
                        x2_sum[k] += (wk[:, None] * (X ** 2)).sum(axis=0)

            # M-step
            self.pi = pi_acc / np.maximum(pi_acc.sum(), self.eps)

            for i in range(self.K):
                if A_den[i] > 0:
                    self.A[i] = A_num[i] / np.maximum(A_den[i], self.eps)
                # keep it normalized no matter what
            self.A = normalize_rows(np.maximum(self.A, self.eps), self.eps)

            for k in range(self.K):
                if w_sum[k] > 0:
                    mu = x_sum[k] / np.maximum(w_sum[k], self.eps)
                    ex2 = x2_sum[k] / np.maximum(w_sum[k], self.eps)
                    var = ex2 - mu ** 2
                    var = np.maximum(var, self.var_floor)
                    self.means[k] = mu
                    self.vars_[k] = var

            ll_hist.append(total_ll)

            # stop early if not improving
            if last_ll is not None and abs(total_ll - last_ll) < tol:
                break
            last_ll = total_ll

        return ll_hist

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior state probabilities gamma(T,K)."""
        _, gamma, _, _ = self._forward_backward(X)
        return gamma

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely state path, standard Viterbi in log space."""
        T = X.shape[0]
        logB = gaussian_logpdf_diag(X, self.means, self.vars_)
        logA = np.log(np.maximum(self.A, self.eps))
        logpi = np.log(np.maximum(self.pi, self.eps))

        delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)

        delta[0] = logpi + logB[0]
        psi[0] = 0

        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA  # (K,K)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + np.max(scores, axis=0)

        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def reorder_states_by_feature(self, feature_index: int, descending: bool = True) -> np.ndarray:
        """
        After unsupervised training, states are permuted.
        We order states by emission mean of one chosen feature.
        For welfare: rumination_z tends to be higher in normal -> descending=True.
        """
        key = self.means[:, feature_index]
        order = np.argsort(key)
        if descending:
            order = order[::-1]

        self.pi = self.pi[order]
        self.A = self.A[order][:, order]
        self.means = self.means[order]
        self.vars_ = self.vars_[order]
        return order