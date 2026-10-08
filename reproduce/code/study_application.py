# -*- coding: utf-8 -*-
r"""End-to-end application of Algorithm 1 to the GaAs lasers.

The question a user meets in practice: a model is fitted to records kept on one
inspection schedule and then run on another.  Both parameterisations are fitted
identically at the reference schedule; they differ only in how they move to the
deployment schedule.  Everything is judged against what the held-out unit did.

Protocol (leave one unit out, 15 folds)
  fit      pooled within-unit squared CV of the increments at h_ref, from the 14
           other units (each unit's own rate is divided out first);
  deploy   the held-out unit is inspected every h_dep hours.  At an origin t_k
           its rate is estimated from its own record, x_k / t_k, and Algorithm 1
           gives the law of the steps on the h_dep grid:
             consistent  mean rho*h_dep, squared CV  cv2_ref * h_ref / h_dep
             interval    mean rho*h_dep, squared CV  cv2_ref   (fixed shape)
  check    (a) the level at 4000 h: coverage of the 80 % and 90 % intervals;
           (b) failure (10 % current increase) by 4000 h: predicted probability
               against the three observed failures;
           (c) the replacement rule of Section 7.1: replace at the first
               inspection where the 5 % RUL quantile is below one interval.

Two directions: coarse records (1000 h) deployed on a fine schedule (250 h),
where interval scaling is too confident, and the reverse, where it is too
cautious.  BS and Weibull steps are both run.

The data come from the SMRD package and are not redistributed here.

    python code/study_application.py

Writes results/realdata/application.json.
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
from study_weibull_check import nu_of                             # noqa: E402
from scipy import special                                         # noqa: E402

OUT = os.path.join(ROOT, "results", "realdata")
RNG = np.random.default_rng(int(os.environ.get("APP_SEED", 20261007)))
NSIM = 20_000
D_FAIL = 10.0
T_END = 4000.0
ETA = 0.05
ORIGINS = (1000.0, 2000.0, 3000.0)
THRESHOLDS = (6.0, 7.0, 8.0, 9.0, 10.0)      # % increase; 10 % is failure
DIRECTIONS = ((1000.0, 250.0), (250.0, 1000.0))       # (h_ref, h_dep)


PARAM_UNC = os.environ.get("APP_PARAM", "1") == "1"   # carry cv2 uncertainty
NBOOT = 4000


def bs_steps(mean, cv2, size):
    r = np.minimum(cv2, 4.99)                          # Algorithm 1: r < 5
    u = np.sqrt(1 + 3 * r)
    A = 2 * r * (u + 4) / ((5 - r) * (u + 1))
    beta = mean / (1 + A / 2)
    z = RNG.standard_normal(size)
    a = np.sqrt(A)
    return beta * (a * z / 2 + np.sqrt((a * z / 2) ** 2 + 1)) ** 2


def weib_steps(mean, cv2, size):
    nu = nu_of(np.asarray(cv2, float))
    lam = mean / special.gamma(1 + 1 / nu)
    return lam * (-np.log(RNG.random(size))) ** (1 / nu)


def gamma_steps(mean, cv2, size):
    """Gamma steps with the given mean and squared CV.  In the consistent form
    this is the gamma process itself: the exact, Levy, baseline."""
    k = 1.0 / np.asarray(cv2, float)
    return RNG.gamma(k, mean / k, size)


def ig_steps(mean, cv2, size):
    """Inverse Gaussian steps; in the consistent form, the IG process."""
    lam = mean / np.asarray(cv2, float)
    return RNG.wald(np.broadcast_to(mean, size), np.broadcast_to(lam, size))


LAWS = {"BS": bs_steps, "Weibull": weib_steps, "gamma": gamma_steps, "IG": ig_steps}


def on_grid(t, x, h):
    keep = np.isclose(np.mod(t, h), 0) | np.isclose(np.mod(t, h), h)
    return t[keep], x[keep]


def cv2_pooled(P, h, with_df=False):
    """Within-unit squared CV at step h, each unit's rate divided out."""
    ss, df = 0.0, 0
    for t, x in P:
        tt, xx = on_grid(t, x, h)
        y = np.diff(xx)
        z = y / y.mean()
        ss += np.sum((z - 1) ** 2)
        df += len(z) - 1
    return (ss / df, df) if with_df else ss / df


def simulate(law, x_k, t_k, h, cv2, n_steps, df=None):
    """Paths from (t_k, x_k).  Two parameter uncertainties are carried into
    every path.  The rate x_k/t_k: under the model's own variance claim on this
    grid, x_k has squared CV cv2*h/t_k (lognormal, mean preserved).  The squared
    CV itself, estimated from df degrees of freedom: cv2 * df / chi2_df, the
    scaled-inverse-chi-square posterior under a flat prior on log cv2."""
    if df is not None and PARAM_UNC:
        cv2 = cv2 * df / RNG.chisquare(df, (NSIM, 1))
    sd = np.sqrt(cv2 * h / t_k)
    rho = x_k / t_k * np.exp(sd * RNG.standard_normal((NSIM, 1)) - sd ** 2 / 2)
    dx = LAWS[law](1.0 * h, cv2, (NSIM, n_steps)) * rho
    return x_k + np.cumsum(dx, axis=1)


def first_cross(tt, xx, D):
    hit = np.nonzero(xx >= D)[0]
    return float(tt[hit[0]]) if len(hit) else np.inf


def replace_time(law, tt, xx, h, cv2, D, T_obs, df=None):
    """Section 7.1 rule, run online: replace at the first inspection (from 500 h)
    at which failure before the next inspection is more likely than ETA."""
    for j in range(1, len(tt) - 1):
        if tt[j] >= T_obs:
            break
        if tt[j] < 500:
            continue
        S = simulate(law, xx[j], tt[j], h, cv2, 1, df)
        if np.mean(S[:, 0] >= D) > ETA:
            return float(tt[j])
    return np.inf


def crps(sample, y):
    """Continuous ranked probability score of a sample forecast (lower is better):
    E|X - y| - E|X - X'| / 2, the second term from the order statistics."""
    x = np.sort(sample)
    n = len(x)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2)


def paired_ci(La, Lb, key):
    """Mean of a - b over forecasts, with a 95 % interval resampling units."""
    units = sorted({r["unit"] for r in La})
    by = {u: [ra[key] - rb[key] for ra, rb in zip(La, Lb) if ra["unit"] == u] for u in units}
    rng = np.random.default_rng(11)
    v = []
    for _ in range(NBOOT):
        pick = rng.choice(units, len(units))
        v.append(np.mean([d for u in pick for d in by[u]]))
    allv = [d for u in units for d in by[u]]
    return [float(np.mean(allv)), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def unit_ci(L, key):
    """95 % interval for a coverage, resampling units (forecasts of one unit
    share its path, so they are not independent)."""
    units = sorted({r["unit"] for r in L})
    by = {u: [r[key] for r in L if r["unit"] == u] for u in units}
    rng = np.random.default_rng(7)
    v = []
    for _ in range(NBOOT):
        pick = rng.choice(units, len(units))
        hits = [h for u in pick for h in by[u]]
        v.append(np.mean(hits))
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def run(law, h_ref, h_dep):
    P = SR.paths(SR.load("gaaslaser"), "unit", "hours", "increase")
    lev = {m: [] for m in ("consistent", "interval")}
    pol = {m: [] for m in ("consistent", "interval")}
    cv2_list = []
    for i, (t, x) in enumerate(P):
        cv2_ref, df = cv2_pooled([p for j, p in enumerate(P) if j != i], h_ref,
                                 with_df=True)
        cv2_list.append(cv2_ref)
        cv2 = {"consistent": cv2_ref * h_ref / h_dep, "interval": cv2_ref}
        tt, xx = on_grid(t, x, h_dep)
        for tk in ORIGINS:
            k = int(np.argmin(np.abs(tt - tk)))
            n = int(round((T_END - tt[k]) / h_dep))
            for m in cv2:
                S = simulate(law, xx[k], tt[k], h_dep, cv2[m], n, df)
                end = S[:, -1]
                lo80, hi80, lo90, hi90 = np.quantile(end, [.1, .9, .05, .95])
                lev[m].append({"unit": i, "origin": tk, "obs": float(xx[-1]),
                               "in80": bool(lo80 <= xx[-1] <= hi80),
                               "in90": bool(lo90 <= xx[-1] <= hi90),
                               "w90": float(hi90 - lo90),
                               "pit": float(np.mean(end <= xx[-1])),
                               "crps": crps(end, xx[-1]),
                               "is90": float(hi90 - lo90 + 20 * max(lo90 - xx[-1], 0)
                                             + 20 * max(xx[-1] - hi90, 0))})
        for D in THRESHOLDS:
            T_obs = first_cross(tt, xx, D)
            if not np.isfinite(T_obs):
                continue
            for m in cv2:
                t_rep = replace_time(law, tt, xx, h_dep, cv2[m], D, T_obs, df)
                pol[m].append({"unit": i, "D": D, "T_obs": T_obs,
                               "t_rep": t_rep, "missed": bool(t_rep > T_obs)})
    res = {"law": law, "h_ref": h_ref, "h_dep": h_dep,
           "cv2_ref_mean": float(np.mean(cv2_list)),
           "crps_diff": paired_ci(lev["consistent"], lev["interval"], "crps"),
           "is90_diff": paired_ci(lev["consistent"], lev["interval"], "is90")}
    for m in ("consistent", "interval"):
        L, Q = lev[m], pol[m]
        res[m] = {
            "cover80": float(np.mean([r["in80"] for r in L])),
            "cover90": float(np.mean([r["in90"] for r in L])),
            "ci80": unit_ci(L, "in80"), "ci90": unit_ci(L, "in90"),
            "width90": float(np.mean([r["w90"] for r in L])),
            "crps": float(np.mean([r["crps"] for r in L])),
            "is90": float(np.mean([r["is90"] for r in L])),
            "pit_extreme": float(np.mean([min(r["pit"], 1 - r["pit"]) < 0.05
                                          for r in L])),
            "n_level": len(L),
            "n_events": len(Q),
            "missed": int(sum(r["missed"] for r in Q)),
            "missed_D10": int(sum(r["missed"] for r in Q if r["D"] == 10)),
            "n_events_D10": int(sum(r["D"] == 10 for r in Q)),
            "mean_lead": float(np.mean([r["T_obs"] - r["t_rep"]
                                        for r in Q if not r["missed"]])),
            "levels": L, "policy": Q,
        }
    return res


def main():
    out = []
    for law in LAWS:
        for h_ref, h_dep in DIRECTIONS:
            r = run(law, h_ref, h_dep)
            out.append(r)
            print(f"{law:8s} ref {h_ref:5.0f} -> dep {h_dep:5.0f}  "
                  f"cv2_ref {r['cv2_ref_mean']:.4f}  CRPS cons-int {np.round(r['crps_diff'],3)} IS90 {np.round(r['is90_diff'],2)}")
            for m in ("consistent", "interval"):
                s = r[m]
                print(f"   {m:10s} cov80 {s['cover80']:.2f} {np.round(s['ci80'],2)} cov90 {s['cover90']:.2f} {np.round(s['ci90'],2)}"
                      f" w90 {s['width90']:.2f} pitx {s['pit_extreme']:.2f}"
                      f"  missed {s['missed']}/{s['n_events']}"
                      f" (D10 {s['missed_D10']}/{s['n_events_D10']})"
                      f" lead {s['mean_lead']:.0f}")
    with open(os.path.join(OUT, "application.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
