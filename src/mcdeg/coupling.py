"""Damage coupling, the first-passage (RUL) law, and Algorithm 1.

The coupled process (Definition 3 of the paper) takes steps of size

    s_k = (dt_k / dt_ref) * q(X_{k-1})^gamma,     q(x) = 1 + x / x_ref,

each drawn from the moment-consistent law of the chosen family, so that a step
has mean s*mu and variance s*sigma^2, with (mu, sigma^2) the reference moments
of one step of length dt_ref at the undamaged state.  gamma = 0 gives the
homogeneous process.  Under Paris--Erdogan crack growth with an indicator
proportional to crack length, gamma = m/2.

The RUL from state x to the threshold D is approximated (Propositions 7 and 8)
by an inverse Gaussian law whose mean and spread accumulate over two different
distances:

    mean   E[tau]  = dt_ref (d_Z + kappa) / mu
    var    Var[tau] = dt_ref^2 sigma^2 d_V / mu^3
    d_Z = psi(D) - psi(x),   d_V = psi_{2 gamma}(D) - psi_{2 gamma}(x),
    psi(x) = int_0^x q(u)^-gamma du.

Use it when about ten or more steps remain and beta/x_ref <= 0.1; otherwise
simulate (predict_rul does this switch, as Algorithm 1 does).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .families import get_family, consistent_params, interval_params

__all__ = ["q", "psi", "distances", "kappa", "rul_ig_params", "ig_cdf", "ig_ppf",
           "step_law", "simulate_first_passage", "predict_rul", "RULPrediction"]


def q(x, x_ref=1.0):
    return 1.0 + np.asarray(x, float) / x_ref


def psi(x, gamma=0.0, x_ref=1.0):
    """Lamperti map psi(x) = int_0^x q(u)^-gamma du (eq. 15)."""
    x = np.asarray(x, float)
    if gamma == 0:
        return x
    if gamma == 1:
        return x_ref * np.log(q(x, x_ref))
    return x_ref / (1 - gamma) * (q(x, x_ref) ** (1 - gamma) - 1)


def kappa(x, D, mu, var, gamma=0.0, x_ref=1.0):
    """Second-order correction of the mean distance (eq. 17)."""
    if gamma == 0:
        return 0.0
    qx, qD = q(x, x_ref), q(D, x_ref)
    return (var / (2 * mu) * (qx ** -gamma - qD ** -gamma)
            + gamma * mu / 2 * np.log(qD / qx))


def distances(x, D, mu, var, gamma=0.0, x_ref=1.0):
    """(d_mu, d_V): the distance over which the mean accumulates, d_Z + kappa,
    and the distance over which the spread accumulates."""
    d_z = psi(D, gamma, x_ref) - psi(x, gamma, x_ref)
    d_v = psi(D, 2 * gamma, x_ref) - psi(x, 2 * gamma, x_ref)
    return d_z + kappa(x, D, mu, var, gamma, x_ref), d_v


def rul_ig_params(x, D, mu, var, gamma=0.0, x_ref=1.0, dt_ref=1.0):
    """(mean, lambda) of the IG approximation to the RUL (eq. 13/19), in time
    units, and n_d, the expected number of remaining steps."""
    d_mu, d_v = distances(x, D, mu, var, gamma, x_ref)
    mean = d_mu * dt_ref / mu
    lam = d_mu ** 3 * dt_ref / (var * d_v)
    return mean, lam, d_mu / mu


def ig_cdf(t, mean, lam):
    """P(T <= t) for T ~ IG(mean, lam) (eq. 14)."""
    return stats.invgauss.cdf(t, mean / lam, scale=lam)


def ig_ppf(p, mean, lam):
    return stats.invgauss.ppf(p, mean / lam, scale=lam)


def step_law(family, mu, var, x_prev, dt=1.0, dt_ref=1.0, gamma=0.0, x_ref=1.0,
             consistent=True):
    """(shape, scale) of the step that starts at state x_prev and lasts dt.
    consistent=False gives interval scaling, for comparison."""
    s = (np.asarray(dt, float) / dt_ref) * q(x_prev, x_ref) ** gamma
    f = consistent_params if consistent else interval_params
    return f(family, mu, var, s)


def simulate_first_passage(family, x0, D, mu, var, dt=1.0, dt_ref=1.0, gamma=0.0,
                           x_ref=1.0, n_paths=20000, max_steps=100000,
                           consistent=True, rng=None):
    """Interpolated first-passage times (from now, in time units) of n_paths
    simulated paths inspected every dt.  Linear interpolation between
    inspections matches the RUL definition of the paper."""
    fam = get_family(family)
    rng = np.random.default_rng(rng)
    x = np.full(n_paths, float(x0))
    tau = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, bool)
    for k in range(max_steps):
        if not alive.any():
            break
        idx = np.nonzero(alive)[0]
        a, b = step_law(fam, mu, var, x[idx], dt, dt_ref, gamma, x_ref, consistent)
        dx = fam.sample(a, b, size=idx.size, rng=rng)
        xn = x[idx] + dx
        hit = xn >= D
        frac = (D - x[idx][hit]) / dx[hit]
        tau[idx[hit]] = (k + frac) * dt
        x[idx] = xn
        alive[idx[hit]] = False
    return tau


@dataclass
class RULPrediction:
    mean: float
    sd: float
    quantile: float          # the eta-quantile of the RUL
    replace_at: float        # t_k + quantile
    method: str              # "IG" or "simulation"
    n_d: float               # expected number of remaining reference steps


def predict_rul(family, x_k, D, mu, var, t_k=0.0, eta=0.05, dt=1.0, dt_ref=1.0,
                gamma=0.0, x_ref=1.0, nd_min=10.0, max_scale_ratio=0.1,
                n_paths=20000, rng=None):
    """Algorithm 1: the RUL law from state x_k at time t_k, and the replacement
    time t_k + Q_eta(RUL).

    The closed-form IG approximation is used only when both validity conditions
    of the paper hold: at least nd_min reference steps remain, and (under
    coupling) the reference step is small against x_ref, mu / x_ref <=
    max_scale_ratio.  Otherwise the consistent process is simulated.  For BS,
    a step whose squared CV reaches 5 raises ValueError (no consistent BS step
    exists on that grid)."""
    if x_k >= D:
        return RULPrediction(0.0, 0.0, 0.0, t_k, "failed", 0.0)
    m, lam, n_d = rul_ig_params(x_k, D, mu, var, gamma, x_ref, dt_ref)
    small_steps = gamma == 0 or mu / x_ref <= max_scale_ratio
    if n_d >= nd_min and small_steps:
        sd = np.sqrt(m ** 3 / lam)
        qe = float(ig_ppf(eta, m, lam))
        return RULPrediction(float(m), float(sd), qe, t_k + qe, "IG", float(n_d))
    tau = simulate_first_passage(family, x_k, D, mu, var, dt, dt_ref, gamma, x_ref,
                                 n_paths=n_paths, rng=rng)
    qe = float(np.quantile(tau, eta))
    return RULPrediction(float(np.mean(tau)), float(np.std(tau)), qe, t_k + qe,
                         "simulation", float(n_d))
