# -*- coding: utf-8 -*-
r"""Explicit error bounds for Propositions 7 and 8 (Theorems B.1 and B.2 of the paper).

Everything is in units of reference steps (Delta t = Delta t_ref), with BS steps
in the moment-consistent parameterisation.  The proofs are in the paper; this
script evaluates the constants they produce and sets them beside the errors
measured by simulation in Tables D.3 and D.4.

Theorem B.1 (homogeneous, gamma = 0; i.i.d. steps X, mean mu, variance sigma^2,
c = sigma^2/mu^2, beta_3 = E|X - mu|^3 / sigma^3, n_d = d/mu):
  mean    -1 <= E[tau] - n_d <= B1 = (sigma^2 + 2 mu^2) / (mu m),  m = E min(X, mu)
  s.d.    |sd(tau) - sqrt(c n_d)| <= B2 = 1/2 + sqrt(E R^2)/mu + sqrt(c) B1 / (2 sqrt(n_d)),
          E R^2 <= (E X^3 + mu E X^2) / m
  law     sup_t |P(tau <= t) - F_IG(t)| <= B3 = max{4c/n_d,
              0.33554 (beta_3 + 0.415) / sqrt(n) + 1/(n sqrt(2 pi e)) + sqrt(c) / (2 sqrt(2 pi n_d))},
          n = max(1, floor(n_d/2 - 1)),  C0 = 0.4748 (Shevtsova).

Theorem B.2 (Paris coupling, any gamma >= 0): with the potential
  V(y) = [psi(y) - sigma^2 q(y)^-gamma / (2 mu) + (gamma mu / 2) ln q(y)] / mu,
  V(D) - V(x) = (d_Z + kappa)/mu is the mean of Proposition 8, and
  |E[tau] - (d_Z + kappa)/mu| <= max{1, V'(D) O_1} + eps * EN,
  EN <= (V(D) - V(x) + V'(D) O_1) / (1 - eps),
  eps = sup_y [gamma(gamma+1) q^-(gamma+2) E[dX^3|y] / (6 mu x_ref^2) + |V''_h(y)| E[dX^2|y] / 2],
  O_1 a bound on the mean crossing step.  A bound on sd(tau) about
  sqrt(sigma^2 d_V / mu^3) follows the same way with U(y) = sigma^2 psi_2gamma(y)/mu^3.

    python code/study_bounds.py

Writes results/theory/bounds.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import integrate, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gbs                                                        # noqa: E402

OUT = os.path.join(ROOT, "results", "theory")
C0 = 0.4748
SQ2PIE = np.sqrt(2 * np.pi * np.e)
PI26 = np.pi ** 2 / 6


# ── BS moments ──────────────────────────────────────────────────────────────
def bs_raw(alpha, beta):
    """Raw moments E T^k, k = 1..4, of BS(alpha, beta)."""
    a2 = alpha ** 2
    return (beta * (1 + a2 / 2),
            beta ** 2 * (1 + 2 * a2 + 1.5 * a2 ** 2),
            beta ** 3 * (1 + 4.5 * a2 + 9 * a2 ** 2 + 7.5 * a2 ** 3),
            beta ** 4 * (1 + 8 * a2 + 30 * a2 ** 2 + 60 * a2 ** 3 + 52.5 * a2 ** 4))


def bs_of_z(z, alpha, beta):
    return beta * (alpha * z / 2 + np.sqrt((alpha * z / 2) ** 2 + 1)) ** 2


def zexp(f):
    """E f(Z), Z standard normal, by adaptive quadrature."""
    return integrate.quad(lambda z: f(z) * stats.norm.pdf(z), -12, 12, limit=400,
                          points=[0.0])[0]


def check_raw():
    for a in (0.2, 0.5, 1.0):
        num = [zexp(lambda z, k=k: bs_of_z(z, a, 1.0) ** k) for k in (1, 2, 3, 4)]
        assert np.allclose(num, bs_raw(a, 1.0), rtol=1e-7), (a, num, bs_raw(a, 1.0))


def abs3(alpha, beta, mu):
    z0 = 2 * (np.sqrt(mu / beta) - np.sqrt(beta / mu)) / alpha    # T(z0) = mu
    f = lambda z: abs(bs_of_z(z, alpha, beta) - mu) ** 3 * stats.norm.pdf(z)
    return integrate.quad(f, -12, z0, limit=400)[0] + integrate.quad(f, z0, 12, limit=400)[0]


def emin(alpha, beta, c):
    """E min(T, c)."""
    zc = 2 * (np.sqrt(c / beta) - np.sqrt(beta / c)) / alpha
    f = lambda z: min(bs_of_z(z, alpha, beta), c) * stats.norm.pdf(z)
    return integrate.quad(f, -12, zc, limit=400)[0] + c * stats.norm.sf(zc)


def tail(alpha, beta, a, k):
    """E[T^k 1{T >= a}]."""
    if a <= 0:
        return bs_raw(alpha, beta)[k - 1]
    za = 2 * (np.sqrt(a / beta) - np.sqrt(beta / a)) / alpha
    if za > 12:
        return 0.0
    return integrate.quad(lambda z: bs_of_z(z, alpha, beta) ** k * stats.norm.pdf(z),
                          za, 14, limit=400)[0]


def consistent(alpha, beta, s):
    mu, var = gbs.bs_mean(alpha, beta), gbs.bs_var(alpha, beta)
    a, b = gbs.bs_consistent_params(alpha, beta, s)
    return float(a), float(b)


# ── Theorem B.1: homogeneous case ─────────────────────────────────────────────
def crossing_moments(alpha, beta):
    """Bounds on E X_N and E X_N^2 for the step that crosses the threshold.

    E X_N^k = E[X^k (U(d) - U(d - X))] <= E[X^k U(X)], U(y) = sum_m P(S_m < y)
    (the count of walk points in [d - x, d) is at most U(x), by restarting at the
    first point in it).  With F(T(z)) = Phi(z) and the Chernoff bound
    P(S_m < x) <= e^{theta x} L^m, L = E e^{-theta X}, for every theta > 0,
        U(x) <= 1 + F(x) + e^{theta x} L^2 / (1 - L),
    and with the truncated-Wald bound U(x) <= (x + c) / E min(X, c) as a check."""
    T = lambda z: bs_of_z(z, alpha, beta)
    ez = lambda f: zexp(f)
    m1, m2, m3, _ = bs_raw(alpha, beta)
    exf = [ez(lambda z, k=k: T(z) ** k * stats.norm.cdf(z)) for k in (1, 2)]
    th_max = 1 / (2 * alpha ** 2 * beta)
    best = [np.inf, np.inf]
    for th in np.geomspace(0.05, 0.9, 40) * th_max:
        L = ez(lambda z: np.exp(-th * T(z)))
        if L >= 1:
            continue
        g = L ** 2 / (1 - L)
        for k, mk in ((1, m1), (2, m2)):
            v = mk + exf[k - 1] + g * ez(lambda z, k=k: T(z) ** k * np.exp(th * T(z)))
            best[k - 1] = min(best[k - 1], v)
    m = emin(alpha, beta, m1)                                # truncated Wald, c = mu
    best[0] = min(best[0], (m2 + m1 * m1) / m)
    best[1] = min(best[1], (m3 + m1 * m2) / m)
    return best


def homogeneous(alpha, n_d):
    beta = 1.0
    m1, m2, m3, _ = bs_raw(alpha, beta)
    mu, var = m1, m2 - m1 ** 2
    sig, c = np.sqrt(var), var / m1 ** 2
    b3 = abs3(alpha, beta, mu) / sig ** 3
    EX1, EX2 = crossing_moments(alpha, beta)          # bounds on E X_N, E X_N^2
    B1, ER2 = EX1 / mu, EX2
    # tau = (d - W)/mu + R(1/mu - 1/X_N), sd((d - W)/mu) = sigma sqrt(EN)/mu
    # (Wald's second identity); |R(1/mu - 1/X_N)| <= |X_N - mu|/mu, E X_N >= mu
    B2 = (np.sqrt(max(ER2 - mu ** 2, 0.0)) / mu
          + np.sqrt(c) * (np.sqrt(n_d + B1) - np.sqrt(n_d)))
    B1 = np.sqrt(max(ER2 - mu ** 2, 0.0)) / mu      # |E tau - n_d| <= E|X_N - mu|/mu
    n = max(1, int(np.floor(n_d / 2 - 1)))
    # Shevtsova (2011), Theorem 2: Delta_n <= 0.33554 (beta_3 + 0.415) / sqrt(n)
    B3 = max(4 * c / n_d, 0.33554 * (b3 + 0.415) / np.sqrt(n) + 1 / (n * SQ2PIE)
             + np.sqrt(c) / (2 * np.sqrt(2 * np.pi * n_d)))
    sd_pred = np.sqrt(c * n_d)
    return {"alpha": alpha, "n_d": n_d, "c": c, "beta3": b3,
            "mean_bound_steps": B1, "mean_bound_pct": 100 * B1 / n_d,
            "sd_bound_steps": B2, "sd_bound_pct": 100 * B2 / sd_pred, "law_bound": B3}


# ── Theorem B.2: Paris coupling ───────────────────────────────────────────────
def coupled(gamma, x_ref, alpha, beta=0.02, D=3.0, x0=0.0, ngrid=600):
    mu, var = gbs.bs_mean(alpha, beta), gbs.bs_var(alpha, beta)
    q = lambda y: 1 + y / x_ref

    def psi(y, g):
        return x_ref / (1 - g) * (q(y) ** (1 - g) - 1) if g != 1 else x_ref * np.log(q(y))

    V = lambda y: (psi(y, gamma) - var / (2 * mu) * q(y) ** -gamma
                   + gamma * mu / 2 * np.log(q(y))) / mu
    Vp = lambda y: (q(y) ** -gamma + var * gamma / (2 * mu * x_ref) * q(y) ** (-gamma - 1)
                    + gamma * mu / (2 * x_ref) / q(y)) / mu
    hpp = lambda y: (var * gamma * (gamma + 1) / (2 * mu * x_ref ** 2) * q(y) ** (-gamma - 2)
                     + gamma * mu / (2 * x_ref ** 2) * q(y) ** -2) / mu            # |h''|
    Vpp = lambda y: gamma * q(y) ** (-gamma - 1) / (mu * x_ref) + hpp(y)          # |V''|
    ys = np.linspace(x0, D, ngrid)
    mom = []
    for y in ys:
        s = q(y) ** gamma
        a, b = consistent(alpha, beta, s)
        mom.append((s, a, b) + bs_raw(a, b))
    mom = np.array(mom)
    s_, a_, b_, M1, M2, M3, M4 = mom.T
    eps_y = (gamma * (gamma + 1) * q(ys) ** (-gamma - 2) * M3 / (6 * mu * x_ref ** 2)
             + 0.5 * hpp(ys) * M2)
    eps = float(eps_y.max())
    # crossing step: cells of width l below D.  A visit to cell j (distance
    # [j l, (j+1) l) from D) crosses only with a step >= j l, and the expected
    # number of steps taken from inside the cell is at most 2l / inf E min(dX, l)
    # over the cell (truncated Wald).  So
    #   E dX_N^k <= sum_j 2l / m_j * sup_{y in cell j} E[dX^k 1{dX >= j l} | y].
    def over(k):
        best = np.inf
        for l in np.array([0.5, 1.0, 2.0]) * M1[-1]:
            tot, j = 0.0, 0
            while D - (j + 1) * l > x0 - l:
                lo_y, hi_y = max(D - (j + 1) * l, x0), D - j * l
                idx = np.nonzero((ys >= lo_y - 1e-12) & (ys <= hi_y + 1e-12))[0]
                if len(idx) == 0:
                    idx = np.array([int(np.argmin(np.abs(ys - hi_y)))])
                t = max(tail(a_[i], b_[i], j * l, k) for i in idx)
                if t > 0:
                    m = min(emin(a_[i], b_[i], l) for i in idx)
                    tot += 2 * l / m * t
                if j > 2 and t < 1e-14:
                    break
                j += 1
            best = min(best, tot)
        return float(best)
    O1, O2 = over(1), over(2)
    mean_pred = V(D) - V(x0)
    EN = (mean_pred + Vp(D) * O1) / (1 - eps)
    mean_err = max(1.0, Vp(D) * O1) + eps * EN
    # s.d.: predicted variance A = sigma^2 d_V / mu^3
    dV = psi(D, 2 * gamma) - psi(x0, 2 * gamma)
    A = var * dV / mu ** 3
    Up_D = var * q(D) ** (-2 * gamma) / mu ** 3
    epsU = float((gamma * var * q(ys) ** (-2 * gamma - 1) / (mu ** 3 * x_ref) * M2).max())
    base = np.sqrt(var) * q(ys) ** (-gamma / 2) / mu
    a_y = gamma / (2 * x_ref) * (var / (mu * q(ys)) + mu * q(ys) ** (gamma - 1))
    eta = base * a_y + 0.5 * Vpp(ys) * np.sqrt(M4)
    zeta = float((2 * base * eta + eta ** 2).max())
    eA = Up_D * O1 + (epsU + zeta) * EN
    EB = Vp(D) * np.sqrt(O2)
    sdM_hi, sdM_lo = np.sqrt(A + eA), np.sqrt(max(A - eA, 0.0))
    sd_hi = (sdM_hi + EB + eps * EN) / (1 - eps) + 0.5
    sd_lo = (sdM_lo - EB - eps * EN) / (1 + eps) - 0.5
    sd_pred = np.sqrt(A)
    return {"gamma": gamma, "x_ref": x_ref, "alpha": alpha, "n_d": mean_pred,
            "eps": eps, "O1": O1, "mean_bound_pct": 100 * mean_err / mean_pred,
            "sd_bound_pct": 100 * max(sd_hi - sd_pred, sd_pred - sd_lo) / sd_pred,
            "sd_pred_steps": float(sd_pred),
            "sd_bound_steps": float(max(sd_hi - sd_pred, sd_pred - sd_lo)),
            "sd_lo": float(sd_lo), "sd_hi": float(sd_hi)}


def main():
    check_raw()
    T2 = json.load(open(os.path.join(OUT, "theory_results.json")))["T2"]
    hom = []
    for i, a in enumerate(T2["alphas"]):
        for j, nd in enumerate(T2["n_d"]):
            r = homogeneous(a, float(nd))
            r["law_measured"] = T2["sup_continuous"][i][j]
            hom.append(r)
            print(f"hom  a={a:.1f} nd={nd:4.0f}  law bound {r['law_bound']:.3f} "
                  f"(measured {r['law_measured']:.4f})  mean {r['mean_bound_pct']:.2f}%"
                  f"  sd {r['sd_bound_pct']:.1f}%  beta3 {r['beta3']:.2f}")
    lam = json.load(open(os.path.join(OUT, "lamperti.json")))["grid"]
    cpl = []
    for g in lam:
        r = coupled(g["gamma"], g["x_ref"], g["alpha"])
        r["mean_measured_pct"] = abs(g["mean_err_pct_2nd"])
        r["sd_measured_pct"] = abs(g["sd_err_pct_2nd"])
        cpl.append(r)
        print(f"cpl  g={g['gamma']:.1f} xr={g['x_ref']:.2f} a={g['alpha']:.1f} nd={r['n_d']:6.1f}"
              f"  eps {r['eps']:.2e}  O1 {r['O1']:.3f}  mean bound {r['mean_bound_pct']:.2f}%"
              f" (meas {r['mean_measured_pct']:.2f}%)  sd bound {r['sd_bound_pct']:.1f}%"
              f" (meas {r['sd_measured_pct']:.2f}%)")
    with open(os.path.join(OUT, "bounds.json"), "w") as f:
        json.dump({"C0": C0, "homogeneous": hom, "coupled": cpl}, f, indent=1)


if __name__ == "__main__":
    main()
