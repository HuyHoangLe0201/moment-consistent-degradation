"""Fitting the reference moments, and the model-free check of the requirement.

Data are a list of paths, each a pair (t, x) of 1-D arrays: inspection times and
degradation readings of one unit, increasing in t.

fit_reference(paths, dt_ref)
    The reference mean and variance of one step of length dt_ref, with each
    unit's own rate divided out.  Works on irregular grids: under the
    consistent model a step of length dt has mean rho*s and variance
    cv2*(rho*s)^2/s, s = dt/dt_ref.

variance_growth_exponent(paths, ...)
    Merge m consecutive increments, measure their within-unit variance, and fit
    Var ~ S^b.  Independent (moment-consistent) increments give b = 1; interval
    scaling, fitted at one step and used at m steps, gives b = 2.  Optional:
      clock       a function t -> tau(t) for decelerating or accelerating paths
                  (e.g. lambda t: t**p), or a per-path list of clock arrays;
      resolution  the reading resolution, whose rounding variance res^2/12 at
                  each end of an increment is removed (Sheppard).
"""
from __future__ import annotations

import numpy as np

__all__ = ["fit_reference", "variance_growth_exponent", "thin"]


def thin(t, x, h, t0=None):
    """Keep the readings that fall on a grid of step h (starting at t0, by
    default the first time)."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    t0 = t[0] if t0 is None else t0
    k = (t - t0) / h
    keep = np.isclose(k, np.round(k), atol=1e-9)
    return t[keep], x[keep]


def fit_reference(paths, dt_ref=1.0, resolution=None):
    """Pooled reference moments of one step of length dt_ref.

    Returns a dict with mu (mean step across units), var, cv2, the per-unit
    rates per dt_ref, and the degrees of freedom of cv2 (for a scaled inverse
    chi-square uncertainty, cv2 * df / chi2_df)."""
    num, df, rates = 0.0, 0, []
    sig2_e = 0.0 if resolution is None else resolution ** 2 / 12
    for t, x in paths:
        t, x = np.asarray(t, float), np.asarray(x, float)
        s = np.diff(t) / dt_ref
        y = np.diff(x)
        if len(y) < 2:
            continue
        rho = y.sum() / s.sum()
        z = y / (rho * s) - 1
        num += np.sum(s * z ** 2 - 2 * sig2_e / (rho ** 2 * s))
        df += len(y) - 1
        rates.append(rho)
    if df == 0:
        raise ValueError("need at least one path with two or more increments")
    cv2 = max(num / df, 1e-12)
    mu = float(np.mean(rates))
    return {"mu": mu, "var": cv2 * mu ** 2, "cv2": cv2, "rates": np.array(rates),
            "df": df}


def _block_variance(paths, m, clocks, sig2_e):
    num, df, S_all = 0.0, 0, []
    for (t, x), tau in zip(paths, clocks):
        S = np.diff(tau[::m])
        y = np.diff(np.asarray(x, float)[::m])
        if len(y) < 2:
            continue
        rho = y.sum() / S.sum()
        num += np.sum((y - rho * S) ** 2 - 2 * sig2_e) / rho ** 2
        df += len(y) - 1
        S_all.extend(S)
    return num / df, float(np.mean(S_all))


def variance_growth_exponent(paths, ms=(1, 2, 3, 4), clock=None, resolution=None,
                             n_boot=2000, rng=None):
    """Exponent b of Var(m-step increment) ~ S^b, with a bootstrap over units.

    Returns a dict with b, a 95 % interval (b_lo, b_hi), and the variances and
    mean clock lengths per m.  b near 1 is what moment consistency requires;
    b = 2 is what interval scaling implies."""
    paths = [(np.asarray(t, float), np.asarray(x, float)) for t, x in paths]
    if clock is None:
        clocks = [t for t, _ in paths]
    elif callable(clock):
        clocks = [clock(t) for t, _ in paths]
    else:
        clocks = [np.asarray(c, float) for c in clock]
    sig2_e = 0.0 if resolution is None else resolution ** 2 / 12

    def fit(P, C):
        V, S = zip(*(_block_variance(P, m, C, sig2_e) for m in ms))
        V = np.maximum(np.asarray(V), 1e-300)
        return float(np.polyfit(np.log(S), np.log(V), 1)[0]), V, S

    b, V, S = fit(paths, clocks)
    rng = np.random.default_rng(rng)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(paths), len(paths))
        try:
            boots.append(fit([paths[i] for i in idx], [clocks[i] for i in idx])[0])
        except (ZeroDivisionError, ValueError, FloatingPointError):
            continue
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan))
    return {"b": b, "b_lo": float(lo), "b_hi": float(hi), "m": list(ms),
            "var": list(map(float, V)), "clock_length": list(map(float, S))}
