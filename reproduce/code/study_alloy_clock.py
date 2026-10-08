# -*- coding: utf-8 -*-
r"""Variance growth of the Alloy-A increments on the damage clock.

study_realdata.py measures the growth exponent b on calendar time and has to
remove the Paris acceleration by polynomial detrending; the verdict then
depends on the degree.  Here the acceleration is removed by the model itself.
On the fine grid, with gamma from the data, each step gets the clock increment

    s_j = q(x_j)^gamma,           q(x) = 1 + x / x_ref,

and a block of m steps the clock increment S = sum of its s_j.  Under any
increment process on this clock the block increment has mean mu_i S and
variance proportional to S^b with b = 1 (moment consistency); fixed-shape
scaling gives b = 2.  The readings are rounded to 0.01 inch, so each observed
block increment also carries the rounding of its two end readings, variance
2 * RES^2 / 12, which does not grow with S: the Sheppard correction removes it.

    python code/study_alloy_clock.py

Writes results/realdata/alloy_clock.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_realdata as SR                                       # noqa: E402
from study_application_alloy import RES, SIG2_E, fit, q            # noqa: E402

OUT = os.path.join(ROOT, "results", "realdata")
MS = (1, 2, 3, 4)
NBOOT = 4000


def blocks(P, gamma, m):
    """(unit, S, y) for non-overlapping blocks of m fine steps."""
    rows = []
    for i, (t, x) in enumerate(P):
        s = q(x[:-1]) ** gamma
        y = np.diff(x)
        for a in range(0, len(y) - m + 1, m):
            rows.append((i, s[a:a + m].sum(), y[a:a + m].sum()))
    return rows


def exponent(P, gamma, sheppard):
    """Slope of log V_m on log mean S_m, V_m the pooled within-unit variance of
    the block increments around mu_i S, each unit's level divided out."""
    lv, ls = [], []
    for m in MS:
        R = blocks(P, gamma, m)
        num, df, Ss = 0.0, 0, []
        for i in {r[0] for r in R}:
            S = np.array([r[1] for r in R if r[0] == i])
            y = np.array([r[2] for r in R if r[0] == i])
            if len(y) < 2:
                continue
            mu = y.sum() / S.sum()
            num += np.sum((y - mu * S) ** 2 - sheppard * 2 * SIG2_E) / mu ** 2
            df += len(y) - 1
            Ss += list(S)
        lv.append(np.log(max(num / df, 1e-12)))
        ls.append(np.log(np.mean(Ss)))
    return float(np.polyfit(ls, lv, 1)[0])


def main():
    P = SR.paths(SR.load("alloya"), "specimen", "megacycles", "inches")
    gamma = fit(P, 0.01)["gamma"]
    rng = np.random.default_rng(20261008)
    out = {"gamma": gamma, "res": RES}
    for key, sh in (("raw", 0), ("sheppard", 1)):
        b = exponent(P, gamma, sh)
        bs = []
        for _ in range(NBOOT):
            Pb = [P[j] for j in rng.integers(0, len(P), len(P))]
            bs.append(exponent(Pb, fit(Pb, 0.01)["gamma"], sh))
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out[key] = {"b": b, "b_lo": float(lo), "b_hi": float(hi)}
        print(f"{key:9s} b = {b:.2f}  [{lo:.2f}, {hi:.2f}]")
    with open(os.path.join(OUT, "alloy_clock.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
