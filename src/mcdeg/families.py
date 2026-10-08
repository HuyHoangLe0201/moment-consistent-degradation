"""Two-parameter positive families and their moment-consistent parameterisation.

A family is a scale family in its scale parameter: a variate with shape theta
and scale lam has mean lam*mu(theta) and variance lam^2*sigma^2(theta), so its
squared coefficient of variation g(theta) = sigma^2/mu^2 does not depend on lam.

Interval scaling keeps the shape and lets the scale absorb the interval,
    dX ~ P(theta, lam * s),   s = dt / dt_ref.
The accumulated variance then shrinks with the grid (Proposition 1 of the paper).

The moment-consistent parameterisation (Proposition 2) instead asks for
    E[dX] = s * m,   Var[dX] = s * v,
with (m, v) the reference moments at dt_ref.  Dividing the two conditions leaves
    g(theta_s) = r = v / (m^2 s),
which has a solution exactly when r lies in the range of g, and a unique one
when g is one-to-one.  Every family below has a strictly monotone g; the
Birnbaum--Saunders range is (0, 5), the others (0, inf).

Each family exposes:
    g(shape)                       squared coefficient of variation
    shape_from_cv2(r)              the inverse of g
    from_moments(mean, var)        (shape, scale) with these moments
    moments(shape, scale)          (mean, var)
    sample(shape, scale, size, rng)
and the module-level helpers consistent_params() and interval_params() turn a
reference (mean, var) and an interval ratio s into the parameters of one step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import special

__all__ = ["Family", "BS", "WEIBULL", "GAMMA", "IG", "LOGNORMAL", "FAMILIES",
           "get_family", "consistent_params", "interval_params", "step_moments",
           "BS_CV2_MAX"]

BS_CV2_MAX = 5.0


@dataclass(frozen=True)
class Family:
    name: str
    g: Callable
    shape_from_cv2: Callable
    from_moments: Callable
    moments: Callable
    sample: Callable
    cv2_max: float = np.inf          # supremum of the range of g

    def __repr__(self):              # pragma: no cover - cosmetic
        return f"Family({self.name!r})"


def _arr(x):
    return np.asarray(x, dtype=float)


# ── Birnbaum--Saunders ─────────────────────────────────────────────────────
#  shape alpha, scale beta; mean beta(1 + a^2/2), var beta^2 a^2 (1 + 5a^2/4)
def _bs_g(alpha):
    a2 = _arr(alpha) ** 2
    return a2 * (1 + 1.25 * a2) / (1 + 0.5 * a2) ** 2


def _bs_shape_from_cv2(r, clip=True):
    """Closed-form root of g(alpha) = r (Theorem 1), written to avoid
    cancellation at small r.  With clip=True r is kept just below 5, the
    finest-grid boundary, as Algorithm 1 does; otherwise r >= 5 raises."""
    r = _arr(r)
    if clip:
        r = np.minimum(r, BS_CV2_MAX * (1 - 1e-9))
    elif np.any(r >= BS_CV2_MAX):
        raise ValueError("BS: squared CV >= 5 has no root (grid finer than n_max)")
    u = np.sqrt(1 + 3 * r)
    A = 2 * r * (u + 4) / ((5 - r) * (u + 1))
    return np.sqrt(A)


def _bs_moments(alpha, beta):
    a2 = _arr(alpha) ** 2
    beta = _arr(beta)
    return beta * (1 + a2 / 2), beta ** 2 * a2 * (1 + 1.25 * a2)


def _bs_from_moments(mean, var):
    alpha = _bs_shape_from_cv2(_arr(var) / _arr(mean) ** 2)
    return alpha, _arr(mean) / (1 + alpha ** 2 / 2)


def _bs_sample(alpha, beta, size=None, rng=None):
    rng = np.random.default_rng(rng)
    shape = size if size is not None else np.broadcast(_arr(alpha), _arr(beta)).shape
    z = rng.standard_normal(shape)
    h = _arr(alpha) * z / 2
    return _arr(beta) * (h + np.sqrt(h * h + 1)) ** 2


BS = Family("BS", _bs_g, _bs_shape_from_cv2, _bs_from_moments, _bs_moments,
            _bs_sample, cv2_max=BS_CV2_MAX)


# ── Weibull ────────────────────────────────────────────────────────────────
#  shape nu, scale lam; P(W > w) = exp(-(w/lam)^nu)
def _wb_g(nu):
    x = 1.0 / _arr(nu)
    return np.exp(special.gammaln(1 + 2 * x) - 2 * special.gammaln(1 + x)) - 1


_WB_LOGNU_LO, _WB_LOGNU_HI = np.log(0.02), np.log(1e5)


def _wb_shape_from_cv2(r, iters=90):
    """Root of g_W(nu) = r by bisection on log nu.  g_W is strictly decreasing
    with range (0, inf) (Proposition 6); the bracket [0.02, 1e5] covers squared
    CVs from about 1e-10 to 1e29."""
    lr = np.log(_arr(r))
    lo = np.full(lr.shape, _WB_LOGNU_LO)
    hi = np.full(lr.shape, _WB_LOGNU_HI)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = np.log(_wb_g(np.exp(mid))) > lr      # g too big -> nu too small
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return np.exp(0.5 * (lo + hi))


def _wb_moments(nu, lam):
    x = 1.0 / _arr(nu)
    m1 = _arr(lam) * special.gamma(1 + x)
    return m1, m1 ** 2 * _wb_g(nu)


def _wb_from_moments(mean, var):
    nu = _wb_shape_from_cv2(_arr(var) / _arr(mean) ** 2)
    return nu, _arr(mean) / special.gamma(1 + 1 / nu)


def _wb_sample(nu, lam, size=None, rng=None):
    rng = np.random.default_rng(rng)
    shape = size if size is not None else np.broadcast(_arr(nu), _arr(lam)).shape
    return _arr(lam) * (-np.log(rng.random(shape))) ** (1 / _arr(nu))


WEIBULL = Family("Weibull", _wb_g, _wb_shape_from_cv2, _wb_from_moments,
                 _wb_moments, _wb_sample)


# ── gamma ──────────────────────────────────────────────────────────────────
#  shape k, scale c; consistent form = the gamma process (k_s = s k, c_s = c)
GAMMA = Family(
    "gamma",
    g=lambda k: 1.0 / _arr(k),
    shape_from_cv2=lambda r: 1.0 / _arr(r),
    from_moments=lambda mean, var: (_arr(mean) ** 2 / _arr(var), _arr(var) / _arr(mean)),
    moments=lambda k, c: (_arr(k) * _arr(c), _arr(k) * _arr(c) ** 2),
    sample=lambda k, c, size=None, rng=None: np.random.default_rng(rng).gamma(
        _arr(k), _arr(c), size),
)


# ── inverse Gaussian ───────────────────────────────────────────────────────
#  "shape" phi = lambda/mean, "scale" the mean; consistent form = the IG process
IG = Family(
    "IG",
    g=lambda phi: 1.0 / _arr(phi),
    shape_from_cv2=lambda r: 1.0 / _arr(r),
    from_moments=lambda mean, var: (_arr(mean) ** 2 / _arr(var), _arr(mean)),
    moments=lambda phi, mean: (_arr(mean), _arr(mean) ** 2 / _arr(phi)),
    sample=lambda phi, mean, size=None, rng=None: np.random.default_rng(rng).wald(
        np.broadcast_to(_arr(mean), size if size is not None else np.shape(mean)),
        np.broadcast_to(_arr(phi) * _arr(mean), size if size is not None else np.shape(mean))),
)


# ── lognormal ──────────────────────────────────────────────────────────────
#  shape sigma, scale exp(mu); g = exp(sigma^2) - 1
LOGNORMAL = Family(
    "lognormal",
    g=lambda s: np.expm1(_arr(s) ** 2),
    shape_from_cv2=lambda r: np.sqrt(np.log1p(_arr(r))),
    from_moments=lambda mean, var: (np.sqrt(np.log1p(_arr(var) / _arr(mean) ** 2)),
                                    _arr(mean) / np.sqrt(1 + _arr(var) / _arr(mean) ** 2)),
    moments=lambda s, sc: (_arr(sc) * np.exp(_arr(s) ** 2 / 2),
                           (_arr(sc) * np.exp(_arr(s) ** 2 / 2)) ** 2 * np.expm1(_arr(s) ** 2)),
    sample=lambda s, sc, size=None, rng=None: _arr(sc) * np.exp(
        _arr(s) * np.random.default_rng(rng).standard_normal(
            size if size is not None else np.broadcast(_arr(s), _arr(sc)).shape)),
)

FAMILIES = {f.name.lower(): f for f in (BS, WEIBULL, GAMMA, IG, LOGNORMAL)}


def get_family(family):
    """Accept a Family or its name ('BS', 'Weibull', 'gamma', 'IG', 'lognormal')."""
    if isinstance(family, Family):
        return family
    try:
        return FAMILIES[str(family).lower()]
    except KeyError:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")


def step_moments(mean_ref, var_ref, s, consistent=True):
    """Mean and variance of one step of size s (s = dt/dt_ref, times q(x)^gamma
    under damage coupling).  Consistent: (s m, s v).  Interval scaling: (s m, s^2 v)."""
    s = _arr(s)
    return s * mean_ref, (s if consistent else s ** 2) * var_ref


def consistent_params(family, mean_ref, var_ref, s):
    """(shape, scale) of a moment-consistent step of size s (Proposition 2)."""
    fam = get_family(family)
    m, v = step_moments(mean_ref, var_ref, s, True)
    return fam.from_moments(m, v)


def interval_params(family, mean_ref, var_ref, s):
    """(shape, scale) under interval scaling: shape fixed, scale proportional to s."""
    fam = get_family(family)
    m, v = step_moments(mean_ref, var_ref, s, False)
    return fam.from_moments(m, v)
