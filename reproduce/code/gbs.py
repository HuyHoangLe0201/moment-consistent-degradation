"""
Birnbaum-Saunders and inverse Gaussian helpers for the reproduction scripts of
paper 1 (inspection-schedule consistency).  This is the subset of the authors'
working library that those scripts use; the reusable implementation of the
method is the mcdeg package in src/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats

SQRT2PI = np.sqrt(2.0 * np.pi)
LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)


# ══════════════════════════════════════════════════════════════════════════
# 1.  Birnbaum-Saunders distribution
# ══════════════════════════════════════════════════════════════════════════
def xi(u):
    """xi(u) = sqrt(u) - 1/sqrt(u)."""
    s = np.sqrt(u)
    return s - 1.0 / s


def xi_prime(u):
    """d/du xi(u) = (u^{-1/2} + u^{-3/2}) / 2."""
    return 0.5 * (u ** -0.5 + u ** -1.5)


def bs_pdf(t, alpha, beta):
    """f(t) = xi'(t/beta) / (alpha*beta) * phi(xi(t/beta)/alpha).

    NOTE the normalising constant is alpha*beta, not 2*alpha*beta: xi is a
    strictly increasing bijection (0,inf) -> R, so the change of variable has a
    single branch.  Combined with xi'(u) = (u^{-1/2}+u^{-3/2})/2 this recovers
    the textbook form  [(t/b)^{-1/2}+(t/b)^{-3/2}] / (2*a*b*sqrt(2pi)) * exp(.).
    """
    t = np.asarray(t, dtype=float)
    u = t / beta
    return xi_prime(u) / (alpha * beta) * np.exp(-0.5 * (xi(u) / alpha) ** 2) / SQRT2PI


def bs_logpdf(t, alpha, beta):
    """log f_BS(t; alpha, beta); -inf outside the support."""
    t = np.asarray(t, dtype=float)
    out = np.full(np.broadcast(t, alpha, beta).shape, -np.inf, dtype=float)
    ok = t > 0
    if not np.any(ok):
        return out
    u = np.where(ok, t, 1.0) / beta
    z = xi(u) / alpha
    out = np.where(
        ok,
        np.log(xi_prime(u)) - np.log(alpha * beta) - 0.5 * z ** 2 - LOG_SQRT2PI,
        -np.inf,
    )
    return out


def bs_cdf(t, alpha, beta):
    t = np.asarray(t, dtype=float)
    return np.where(t > 0, stats.norm.cdf(xi(np.where(t > 0, t, 1.0) / beta) / alpha), 0.0)


def bs_ppf(p, alpha, beta):
    """Closed-form quantile function (Prop. 'Closed-Form Quantile Function')."""
    z = stats.norm.ppf(p)
    return beta / 4.0 * (alpha * z + np.sqrt(4.0 + (alpha * z) ** 2)) ** 2


def bs_rvs(alpha, beta, size=None, rng=None):
    """O(1) exact sampler:  T = (beta/4)[aZ + sqrt(4 + a^2 Z^2)]^2."""
    rng = rng or np.random.default_rng()
    z = rng.standard_normal(size if size is not None else np.shape(alpha))
    return beta / 4.0 * (alpha * z + np.sqrt(4.0 + (alpha * z) ** 2)) ** 2


def bs_mean(alpha, beta):
    return beta * (1.0 + alpha ** 2 / 2.0)


def bs_var(alpha, beta):
    return beta ** 2 * alpha ** 2 * (1.0 + 1.25 * alpha ** 2)


# ── BS-Student-t (generalised BS with g = t_nu) ─────────────────────────────
def ig_cdf(t, mu, lam):
    t = np.asarray(t, dtype=float)
    ok = t > 0
    tt = np.where(ok, t, 1.0)
    a = np.sqrt(lam / tt) * (tt / mu - 1.0)
    b = -np.sqrt(lam / tt) * (tt / mu + 1.0)
    # exp(2 lam/mu) * Phi(b) computed in log space to avoid overflow
    logterm = 2.0 * lam / mu + stats.norm.logcdf(b)
    return np.where(ok, stats.norm.cdf(a) + np.exp(np.clip(logterm, -700, 700)), 0.0)


class IncrementLaw:
    """Degradation increment  dX ~ Law(theta, scale)  where `scale` absorbs
    the time step and (for the Paris-coupled variant) the current state."""

    name = "abstract"
    param_names: tuple[str, ...] = ()

    def sample(self, th, scale, rng):
        raise NotImplementedError

    def mean(self, th, scale):
        raise NotImplementedError

    def var(self, th, scale):
        raise NotImplementedError

    def logpdf(self, dx, th, scale):
        """log density of the increment."""
        raise NotImplementedError


def bs_consistent_params(alpha, beta, s):
    """Grid-consistent (moment-invariant) BS increment parameters.

    The obvious scaling  dX ~ BS(alpha, beta*s)  is NOT a valid increment law:
    BS is a scale family, so BS(alpha, beta*s) = s * BS(alpha, beta) and hence
        E[dX] = s * m,     Var[dX] = s^2 * v.
    Summing n = T/s such increments gives Var[X(T)] = n s^2 v = T v s, which
    vanishes as the inspection grid is refined -- the process depends on how
    often one happens to measure.

    Instead solve for (alpha_s, beta_s) making BOTH moments additive,
        beta_s (1 + alpha_s^2/2)                     = m s,
        beta_s^2 alpha_s^2 (1 + 5 alpha_s^2/4)       = v s,
    with m, v the reference (s = 1) mean and variance.  Writing A = alpha_s^2 and
    r = v / (m^2 s), the ratio of the two equations is the quadratic

        (5/4 - r/4) A^2 + (1 - r) A - r = 0,

    whose discriminant collapses to 16(1 + 3r), giving the closed root

        A = 2 r (u + 4) / [ (5 - r)(u + 1) ],    u = sqrt(1 + 3r),

    written in the form that avoids cancellation at small r.  Then
    beta_s = m s / (1 + A/2).  s = 1
    returns (alpha, beta) exactly.  The first-passage results depend on the
    increments only through m and v, so they become grid-invariant as well.
    """
    m = bs_mean(alpha, beta)
    v = bs_var(alpha, beta)
    s = np.asarray(s, dtype=float)
    r = v / np.maximum(m ** 2 * s, 1e-300)
    # g(A) = A(1+5A/4)/(1+A/2)^2 increases from 0 to 5, so a solution exists only
    # for r < 5: no BS law has a squared coefficient of variation of 5 or more.
    # Steps so short (or couplings so negative) that they would demand one are
    # clamped to the admissible boundary rather than returning a complex root.
    r = np.minimum(r, 5.0 - 1e-9)
    # The quadratic (5/4 - r/4)A^2 + (1-r)A - r = 0 has discriminant
    # 16(1-r)^2 + 16r(5-r) = 16(1+3r) -- the r^2 terms cancel -- so
    #     A = 2(r - 1 + sqrt(1+3r)) / (5 - r).
    # Written that way it subtracts two near-equal quantities and loses six per
    # cent of the answer at r ~ 1e-9, as does solving the quadratic directly.
    # Rationalising with u = sqrt(1+3r) gives r - 1 + u = r(u+4)/(u+1), which
    # has no such subtraction and shows A ~ r as r -> 0:
    u = np.sqrt(1.0 + 3.0 * r)
    A = 2.0 * r * (u + 4.0) / ((5.0 - r) * (u + 1.0))
    A = np.clip(np.nan_to_num(A, nan=1e-12), 1e-12, 1e12)
    return np.sqrt(A), np.maximum(m * s / (1.0 + A / 2.0), 1e-300)


class BSLaw(IncrementLaw):
    """Birnbaum-Saunders increments in the grid-consistent parameterisation."""
    name = "BS"
    param_names = ("alpha", "beta")

    def __init__(self, consistent: bool = True):
        self.consistent = consistent

    def _ab(self, th, scale):
        if self.consistent:
            return bs_consistent_params(th["alpha"], th["beta"], scale)
        return th["alpha"], th["beta"] * scale

    def sample(self, th, scale, rng):
        a, b = self._ab(th, scale)
        return bs_rvs(a, b, size=np.shape(th["alpha"]), rng=rng)

    def mean(self, th, scale):
        a, b = self._ab(th, scale)
        return bs_mean(a, b)

    def var(self, th, scale):
        a, b = self._ab(th, scale)
        return bs_var(a, b)

    def logpdf(self, dx, th, scale):
        a, b = self._ab(th, scale)
        return bs_logpdf(dx, a, b)


def lamperti(x, gamma, x_ref):
    """psi(x) = int_0^x q(u)^{-gamma} du  with  q(u) = 1 + u/x_ref.

    Under the Paris-coupled process dX_k ~ BS(alpha, beta*(dt/dt_ref)*q(X)^gamma)
    the transformed increment satisfies  dZ_k = psi(X_k) - psi(X_{k-1})
                                              = BS(alpha, beta*dt/dt_ref) + O(dX^2),
    so the homogeneous first-passage theory applies on the psi-scale with
    distance-to-failure  d_Z = psi(D) - psi(x_k).
    """
    g = np.asarray(gamma, dtype=float)
    q = 1.0 + np.asarray(x, dtype=float) / x_ref
    near1 = np.abs(g - 1.0) <= 1e-8
    out = np.where(near1,
                   x_ref * np.log(q),
                   x_ref * (np.power(q, 1.0 - np.where(near1, 0.0, g)) - 1.0)
                   / np.where(near1, 1.0, 1.0 - g))
    return out


def lamperti_var_distance(x, D, gamma, x_ref):
    """Effective distance governing the SPREAD of the first-passage time.

    psi maps the coupled process to one with constant drift, but not to one with
    constant step variance.  Under the grid-consistent coupling the increment at
    state x has Var[dX] = q(x)^gamma * var1, so on the Lamperti scale

        Var[dZ] = psi'(x)^2 Var[dX] = q^{-2 gamma} q^{gamma} var1
                = q(x)^{-gamma} var1 ,

    which still depends on x.  Accumulating by the renewal CLT along the path,
    Var[tau] = (var1/mu1^3) int_x^D q(u)^{-2 gamma} du, so the variance has its
    own effective distance -- the same Lamperti integral taken at 2*gamma:

        d_V = psi_{2 gamma}(D) - psi_{2 gamma}(x) .

    At gamma = 0 this equals d_Z and the homogeneous formulas are recovered
    exactly.  Using d_V = d_Z instead, as a first-order treatment does,
    over-predicts the RUL standard deviation by up to 40% at gamma = 1.5.
    """
    out = lamperti(D, 2.0 * gamma, x_ref) - lamperti(x, 2.0 * gamma, x_ref)
    fallback = np.maximum(D - np.asarray(x, dtype=float), 0.0)
    return np.where(np.isfinite(out), out, fallback)


def lamperti_distance(x, D, mu1, var1, gamma, x_ref, second_order=True):
    """Effective distance-to-failure on the Lamperti scale.

    `mu1`, `var1` are the mean and variance of the increment at unit scale, so
    the construction applies to ANY increment law, not only Birnbaum-Saunders.

    First order:   d_Z = psi(D) - psi(x),  psi(x) = int_0^x q(u)^{-gamma} du.

    Second order:  psi is curved, so E[dZ] = psi'(x) E[dX] + (1/2) psi''(x) E[dX^2]
    with psi'(x) = q^{-gamma} and psi''(x) = -gamma q^{-gamma-1}/x_ref.  At state
    x the increment has scale s = q(x)^gamma, and grid consistency makes the mean
    proportional to s but the variance ALSO proportional to s -- not to s^2.  So

        E[dX]   = mu1 q^gamma
        E[dX^2] = Var + mean^2 = var1 q^gamma + mu1^2 q^{2 gamma}

    and the two terms carry different powers of q:

        E[dZ] = mu1 - (gamma/(2 x_ref)) [ var1 q^{-1} + mu1^2 q^{gamma-1} ].

    Giving both terms q^{2 gamma} -- i.e. letting the variance inherit the SQUARE
    of the scale -- is the same mistake the grid-consistency theorem exists to
    rule out.  It agrees at x = 0 and drifts as q(x) grows.

    The transformed drift therefore decays as damage accumulates.  Integrating
    dz / E[dZ] along the path, with int q^{-1} du = x_ref ln q and
    int q^{-gamma-1} du = (x_ref/gamma)[q(x)^-gamma - q(D)^-gamma],

        E[tau] = (1/mu1) [ d_Z + kappa ],
        kappa  = var1/(2 mu1) [ q(x)^-gamma - q(D)^-gamma ]
                 + (gamma mu1/2) ln( q(D)/q(x) ),

    the factor x_ref cancelling from both integrals.  kappa vanishes at
    gamma = 0, recovering the homogeneous process exactly.
    """
    dz = lamperti(D, gamma, x_ref) - lamperti(x, gamma, x_ref)
    fallback = np.maximum(D - np.asarray(x, dtype=float), 0.0)
    dz = np.where(np.isfinite(dz), dz, fallback)
    if not second_order:
        return dz
    qD = 1.0 + D / x_ref
    qx = 1.0 + np.maximum(x, 0.0) / x_ref
    safe_mu = np.maximum(mu1, 1e-300)
    kappa = (0.5 * var1 / safe_mu * (np.power(qx, -gamma) - np.power(qD, -gamma))
             + 0.5 * gamma * mu1 * np.log(qD / qx))
    out = dz + np.where(np.isfinite(kappa), kappa, 0.0)
    return np.where(np.isfinite(out), out, fallback)
