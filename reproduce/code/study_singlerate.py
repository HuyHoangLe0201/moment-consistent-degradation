# -*- coding: utf-8 -*-
r"""Can single-rate data separate the two parameterisations once the scale is coupled?

audit_singlerate.py proves it can: under the Paris coupling the effective scale
varies from step to step at a fixed sampling interval, and the naive and
grid-consistent laws then disagree by a factor $s_k$ in every increment variance.
That is an algebraic statement about the two families.  It does not say how much
data is needed to tell them apart, which is the question a reader with one
archive actually has.

So this measures it.  Paths are simulated from the coupled, grid-consistent
process at a single fixed interval -- the exact situation the papers used to say
could not exhibit the defect.  Both parameterisations are then fitted to the same
paths by maximum likelihood, each free in all three parameters, and the fits are
compared by the likelihood ratio.  Nothing here involves two rates.

Two things are reported.  The separation as a function of sample size, so a
reader can see how many records it takes; and what the naive fit does with its
freedom, which is where the practical damage is: it cannot reproduce a
coefficient of variation that changes over life, so it compromises, and the
compromise misstates the increment spread at both ends of the record.

    python code/study_singlerate.py
"""
from __future__ import annotations

import io
import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "theory", "singlerate.json")

ALPHA = 0.35                      # reference increment shape
#  beta is set so that a record runs about 130 acquisitions to failure, the
#  length of a typical run-to-failure record.  It matters: the coupled process
#  accelerates, so a larger beta crosses the threshold in under ten steps, and
#  the answer to "how many records does it take" would then be measured on
#  records shorter than any archive contains.
BETA = 1e-3
GAMMA, XREF = 1.5, 0.1            # Paris m = 3
DFAIL = 1.0                       # failure threshold, ten reference lengths
KMAX = 400                        # hard stop, far beyond a typical crossing
#  The question is how FEW records it takes, so the range is small: two
#  already separate the two families, and an archive holds fifteen.
SIZES = (1, 2, 3, 5, 10)
REPS = 20                         # independent data sets per size
LOG2PI = math.log(2.0 * math.pi)


def bs_mean(a, b):
    return b * (1.0 + a ** 2 / 2.0)


def bs_var(a, b):
    return (b * a) ** 2 * (1.0 + 1.25 * a ** 2)


def bs_logpdf(x, a, b):
    """Transcribed from the BS density in the manuscript, not imported."""
    x = np.asarray(x, float)
    ok = x > 0
    xs = np.where(ok, x, 1.0) / b
    z = (np.sqrt(xs) - 1.0 / np.sqrt(xs)) / a
    dz = (xs ** -0.5 + xs ** -1.5) / (2.0 * a * b)
    lp = -0.5 * z ** 2 - 0.5 * LOG2PI + np.log(np.maximum(dz, 1e-300))
    return np.where(ok, lp, -1e10)


def consistent(a, b, s):
    """eq:consistent_root, in the cancellation-free form the papers print."""
    mu, var = bs_mean(a, b), bs_var(a, b)
    r = np.minimum(var / (mu ** 2 * np.asarray(s, float)), 5.0 - 1e-9)
    u = np.sqrt(1.0 + 3.0 * r)
    A = 2.0 * r * (u + 4.0) / ((5.0 - r) * (u + 1.0))
    return np.sqrt(A), np.asarray(s, float) * mu / (1.0 + A / 2.0)


def naive(a, b, s):
    return np.full_like(np.asarray(s, float), a), b * np.asarray(s, float)


def simulate(n_rec, rng):
    """Coupled process, grid-consistent increments, ONE fixed interval.

    Each path is run to failure and stopped there, as the archives are.  The
    stop is not a convenience: with gamma above one the coupled drift blows up
    in finite time, which is the intended physics of unstable spall growth and
    meaningless past the threshold, so a path that keeps going overflows rather
    than degrading.  Only increments observed before the crossing are used.
    """
    steps, states = [], []
    for _ in range(n_rec):
        x = 0.0
        for _k in range(KMAX):
            s = (1.0 + x / XREF) ** GAMMA        # dt/dt_ref = 1 throughout
            a_s, b_s = consistent(ALPHA, BETA, s)
            w = float(a_s) * rng.standard_normal() / 2.0
            dx = float(b_s) * (w + math.sqrt(w * w + 1.0)) ** 2
            states.append(x)
            steps.append(dx)
            x += dx
            if x >= DFAIL:
                break
    return np.array(steps), np.array(states)


def nll(theta, steps, states, family):
    la, lb, lg = theta
    a, b, g = math.exp(la), math.exp(lb), math.exp(lg)
    if not (1e-3 < a < 5 and 1e-6 < b < 1 and 1e-3 < g < 6):
        return 1e12
    s = (1.0 + states / XREF) ** g
    a_s, b_s = family(a, b, s)
    lp = bs_logpdf(steps, a_s, b_s)
    v = -float(np.sum(lp))
    return v if np.isfinite(v) else 1e12


def fit(steps, states, family):
    x0 = np.array([math.log(ALPHA), math.log(BETA), math.log(GAMMA)])
    best = None
    for jitter in (0.0, 0.3, -0.3):
        r = minimize(nll, x0 + jitter, args=(steps, states, family),
                     method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return best


rng = np.random.default_rng(20260809)
print("  simulating the coupled process at ONE fixed sampling interval,\n"
      "  fitting both parameterisations to the same paths\n")
print(f"  {'records':>8s} {'increments':>11s} {'mean 2*dLL':>12s} "
      f"{'median':>9s} {'p<0.001':>9s} {'consistent wins':>16s}")

results = {}
for n_rec in SIZES:
    stats, wins, sig = [], 0, 0
    for _ in range(REPS):
        steps, states = simulate(n_rec, rng)
        f_c = fit(steps, states, consistent)
        f_n = fit(steps, states, naive)
        d = 2.0 * (f_n.fun - f_c.fun)            # >0 means consistent fits better
        stats.append(d)
        wins += d > 0
        #  Both models have three free parameters and neither nests the other, so
        #  a chi-square with one degree of freedom is a rough yardstick rather
        #  than an exact null; it is used only to mark "clearly separated".
        sig += d > chi2.ppf(0.999, 1)
    d = np.array(stats)
    results[str(n_rec)] = {
        "records": n_rec, "increments": int(steps.size),
        "mean_2dLL": float(d.mean()), "median_2dLL": float(np.median(d)),
        "frac_consistent_better": wins / REPS,
        "frac_clearly_separated": sig / REPS,
    }
    print(f"  {n_rec:8d} {steps.size:11d} {d.mean():12.1f} {np.median(d):9.1f} "
          f"{sig/REPS:9.0%} {wins/REPS:16.0%}")

#  What the naive fit does with its freedom.  One large data set, so the fitted
#  parameters are not themselves noisy.
steps, states = simulate(10, rng)
f_c, f_n = fit(steps, states, consistent), fit(steps, states, naive)
a_c, b_c, g_c = np.exp(f_c.x)
a_n, b_n, g_n = np.exp(f_n.x)
print(f"\n  fitted on {steps.size} increments from the consistent generator "
      f"(alpha={ALPHA}, beta={BETA}, gamma={GAMMA})")
print(f"    consistent fit: alpha {a_c:.4f}  beta {b_c:.5f}  gamma {g_c:.4f}")
print(f"    naive fit:      alpha {a_n:.4f}  beta {b_n:.5f}  gamma {g_n:.4f}")

print("\n  and what that costs, as a function of accumulated damage")
print(f"    {'quantile of x':>14s} {'true sd of step':>16s} {'naive fit sd':>14s} {'error':>9s}")
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
bias = {}
for q in qs:
    x = float(np.quantile(states, q))
    s_true = (1.0 + x / XREF) ** GAMMA
    sd_true = math.sqrt(float(bs_var(*consistent(ALPHA, BETA, s_true))))
    s_fit = (1.0 + x / XREF) ** g_n
    sd_naive = math.sqrt(float(bs_var(*naive(a_n, b_n, s_fit))))
    bias[f"q{int(q*100)}"] = {"x": x, "sd_true": sd_true, "sd_naive": sd_naive,
                              "rel_err": sd_naive / sd_true - 1.0}
    print(f"    {q:14.0%} {sd_true:16.5g} {sd_naive:14.5g} "
          f"{100*(sd_naive/sd_true - 1):+8.1f}%")

payload = {"config": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
                      "x_ref": XREF, "threshold": DFAIL, "reps": REPS},
           #  the fit behind step_sd_bias is its own draw, so its size has to
           #  be recorded rather than borrowed from the last row of by_size
           "fit_increments": int(steps.size),
           "by_size": results,
           "fitted": {"consistent": [a_c, b_c, g_c], "naive": [a_n, b_n, g_n]},
           "step_sd_bias": bias}
io.open(OUT, "w", encoding="utf-8", newline="").write(json.dumps(payload, indent=1))
print(f"\n  wrote results/theory/singlerate.json")
