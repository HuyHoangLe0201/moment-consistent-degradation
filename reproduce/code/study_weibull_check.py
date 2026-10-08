# -*- coding: utf-8 -*-
r"""Do the first-passage approximations hold for Weibull increments too?

The IG approximation (Proposition 7) and the second-order Lamperti reduction
(Proposition 8) use only the first two moments of a step, so their predictions
are the same for any moment-consistent law with the same moments.  Their
errors are not: they depend on the higher moments.  This repeats the two
checks of the paper with Weibull steps whose mean and variance equal those of
the BS steps used there, and stores the results beside the BS ones.

  IG    homogeneous walk, alpha in {0.2,...,0.6} (as squared CV), n_d in
        {5,...,100}; sup distance of the interpolated crossing instant
  LAMP  Paris-coupled walk, the sixteen settings of the Lamperti table

Weibull steps with a state-dependent squared CV are drawn by inverting a
tabulated g_W on a fine logarithmic grid, then W = lambda (-ln U)^(1/nu).

    python code/study_weibull_check.py  ->  results/theory/weibull_check.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gbs                                                        # noqa: E402

OUT = os.path.join(HERE, "..", "results", "theory", "weibull_check.json")
RNG = np.random.default_rng(57)

#  g_W(nu) is strictly decreasing; tabulate log g against log nu and invert
_NU = np.logspace(np.log10(0.05), np.log10(400.0), 6000)
_G = special.gamma(1 + 2 / _NU) / special.gamma(1 + 1 / _NU) ** 2 - 1
_LG, _LN = np.log(_G[::-1]), np.log(_NU[::-1])


def nu_of(cv2):
    return np.exp(np.interp(np.log(cv2), _LG, _LN))


def weibull_steps(mean, cv2, size=None):
    nu = nu_of(np.asarray(cv2, float))
    lam = mean / special.gamma(1 + 1 / nu)
    u = RNG.random(size if size is not None else np.shape(nu))
    return lam * (-np.log(u)) ** (1 / nu)


def ecdf_sup(tau, grid, F):
    ts = np.sort(tau)
    return float(np.max(np.abs(np.searchsorted(ts, grid, side="right") / len(ts) - F)))


def ig_check():
    alphas, nds = [0.2, 0.3, 0.4, 0.5, 0.6], [5, 10, 25, 50, 100]
    tab = np.zeros((len(alphas), len(nds)))
    for i, a in enumerate(alphas):
        m, v = float(gbs.bs_mean(a, 1.0)), float(gbs.bs_var(a, 1.0))
        for j, nd in enumerate(nds):
            d = nd * m
            nmax, n_sim = int(4 * nd + 40), 60_000
            taus = []
            for lo in range(0, n_sim, 20_000):
                dx = weibull_steps(m, v / m ** 2, size=(20_000, nmax))
                S = np.cumsum(dx, axis=1)
                N = np.argmax(S >= d, axis=1)
                r = np.arange(len(N))
                Sprev = np.where(N > 0, S[r, np.maximum(N - 1, 0)], 0.0)
                taus.append(N + (d - Sprev) / dx[r, N])
            tau = np.concatenate(taus)
            mu_r, lam_r = d / m, d ** 2 / v
            grid = np.linspace(0.25 * mu_r, 2.6 * mu_r, 300)
            tab[i, j] = ecdf_sup(tau, grid, gbs.ig_cdf(grid, mu_r, lam_r))
        print("IG  alpha-equiv", a, np.round(tab[i], 4))
    return {"alphas": alphas, "n_d": nds, "sup_continuous": tab.tolist()}


def lamperti_check():
    rows = []
    for gamma, x_ref in [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0),
                         (1.5, 0.4), (1.5, 2.5), (1.0, 0.25)]:
        for alpha in (0.2, 0.4):
            beta, dt, x0, D, n_sim = 0.02, 1.0, 0.0, 3.0, 40_000
            m, v = float(gbs.bs_mean(alpha, beta)), float(gbs.bs_var(alpha, beta))
            x = np.full(n_sim, x0)
            tau = np.zeros(n_sim)
            alive = np.ones(n_sim, bool)
            for _ in range(4000):
                if not alive.any():
                    break
                s = (1.0 + np.maximum(x, 0.0) / x_ref) ** gamma
                dx = weibull_steps(m * s, (v / m ** 2) / s)
                xn = x + dx
                cross = alive & (xn >= D)
                frac = np.where(cross, (D - x) / np.maximum(dx, 1e-300), 1.0)
                tau = np.where(alive, tau + dt * np.clip(frac, 0, 1), tau)
                x = np.where(alive, xn, x)
                alive &= ~cross
            dV = float(gbs.lamperti_var_distance(x0, D, gamma, x_ref))
            out = {"gamma": gamma, "x_ref": x_ref, "alpha": alpha}
            for order, tag in ((False, "1st"), (True, "2nd")):
                dZ = float(gbs.lamperti_distance(x0, D, m, v, gamma, x_ref, second_order=order))
                mu = dZ * dt / m
                lam = dZ ** 3 * dt / (v * (dV if order else dZ))
                grid = np.linspace(0.3 * mu, 2.2 * mu, 300)
                out["sup_" + tag] = ecdf_sup(tau, grid, gbs.ig_cdf(grid, mu, lam))
                out["mean_err_pct_" + tag] = float(100 * (tau.mean() / mu - 1))
                if order:
                    #  predicted over simulated, minus one, as in the BS table
                    out["sd_overpred_pct_dV"] = float(100 * (np.sqrt(mu ** 3 / lam) / tau.std() - 1))
            rows.append(out)
            print("LAMP", gamma, x_ref, alpha, {k: round(v, 4) for k, v in out.items()
                                                 if k.startswith(("sup", "mean", "sd"))})
    return rows


if __name__ == "__main__":
    res = {"ig": ig_check(), "lamperti": lamperti_check()}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("  ->", OUT)
