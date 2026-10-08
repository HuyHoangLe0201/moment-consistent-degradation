"""IG approximation to the multi-step RUL (Table D.4 of the paper).

  T1  first passage of i.i.d. increments -> BS, against the Berry-Esseen bound
  T2  IG approximation to the RUL: continuous crossing instant vs lattice index

T1 is kept because it draws from the same random stream before T2; running it
first reproduces the numbers of the paper exactly (seed 2026).

    python code/study_igapprox.py

Writes results/theory/theory_results.json ({"T1": ..., "T2": ...}).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

import gbs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "theory")
os.makedirs(RES, exist_ok=True)
OUT: dict = {}


def t1_derivation(rng):
    print("T1  BS derivation / Berry-Esseen")
    mu_y, sd_y = 1.0, 0.35
    shape_g = (mu_y / sd_y) ** 2
    rho3 = stats.gamma.expect(lambda x: abs(x - shape_g) ** 3,
                              args=(shape_g,)) / shape_g ** 1.5
    rows = []
    for omega in [25.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
        alpha = sd_y / np.sqrt(mu_y * omega)
        beta = omega / mu_y
        nmax = int(2.2 * beta + 60)
        n_sim, chunk = 200_000, max(1, int(3e7 // nmax))
        Ns = []
        for lo in range(0, n_sim, chunk):
            m = min(chunk, n_sim - lo)
            Y = rng.gamma(shape_g, sd_y ** 2 / mu_y, size=(m, nmax)).astype(np.float32)
            S = np.cumsum(Y, axis=1)
            Ns.append(np.argmax(S >= omega, axis=1) + 1.0)
            del Y, S
        N = np.concatenate(Ns)
        ns = np.unique(np.linspace(0.55 * beta, 1.8 * beta, 90).astype(int))
        ns = ns[(ns >= 1) & (ns < nmax)]
        emp = np.array([(N <= n).mean() for n in ns])
        th = stats.norm.cdf(gbs.xi(ns / beta) / alpha)
        sup = float(np.max(np.abs(emp - th)))
        be = float(0.4748 * rho3 / np.sqrt(omega / mu_y))
        rows.append({"omega_over_mu": omega / mu_y, "alpha": alpha,
                     "sup_error": sup, "be_bound": be, "holds": sup <= be})
        print(f"    w/mu={omega/mu_y:6.0f}  alpha={alpha:.4f}  sup={sup:.4f}  BE={be:.4f}")
    OUT["T1"] = {"rho3_over_sigma3": rho3, "rows": rows}


def t2_ig_approx(rng):
    print("T2  IG approximation to the multi-step RUL")
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    nds = [5, 10, 25, 50, 100]
    tab_c = np.zeros((len(alphas), len(nds)))
    tab_l = np.zeros_like(tab_c)
    for i, a in enumerate(alphas):
        for j, nd in enumerate(nds):
            b = 1.0
            d = nd * gbs.bs_mean(a, b)
            nmax = int(4 * nd + 40)
            n_sim = 60_000
            dx = gbs.bs_rvs(a, b, size=(n_sim, nmax), rng=rng)
            S = np.cumsum(dx, axis=1)
            N = np.argmax(S >= d, axis=1)
            Sprev = np.where(N > 0, S[np.arange(n_sim), np.maximum(N - 1, 0)], 0.0)
            tau = N + (d - Sprev) / dx[np.arange(n_sim), N]
            mu_r, lam_r = d / gbs.bs_mean(a, b), d ** 2 / gbs.bs_var(a, b)
            grid = np.linspace(0.25 * mu_r, 2.6 * mu_r, 300)
            F = gbs.ig_cdf(grid, mu_r, lam_r)
            tab_c[i, j] = np.max(np.abs(np.array([(tau <= g).mean() for g in grid]) - F))
            tab_l[i, j] = np.max(np.abs(np.array([(N + 1.0 <= g).mean() for g in grid]) - F))
        print(f"    alpha={a}: continuous " + " ".join(f"{v:.4f}" for v in tab_c[i]))
    OUT["T2"] = {"alphas": alphas, "n_d": nds,
                 "sup_continuous": tab_c.tolist(), "sup_lattice": tab_l.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    t1_derivation(rng)
    t2_ig_approx(rng)
    with open(os.path.join(RES, "theory_results.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print("wrote results/theory/theory_results.json")
