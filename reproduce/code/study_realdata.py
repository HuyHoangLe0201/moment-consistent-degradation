# -*- coding: utf-8 -*-
r"""Real degradation data for the inspection-schedule question.

Five data sets from Meeker & Escobar, *Statistical Methods for Reliability Data*
(SMRD R package, github.com/Auburngrads/SMRD), placed in data_external/smrd/.
They are NOT redistributed with this code: obtain them from that repository.

Three are degradation paths and answer the paper's question directly:

  gaaslaser   15 GaAs lasers, % increase in operating current every 250 h
  alloya      21 Alloy-A specimens, fatigue crack length every 0.01 Mcycles
  metalwear   12 specimens at three loads, wear depth on an irregular,
              roughly logarithmic grid of 2 ... 500 cycles

Two are not paths and cannot carry an increment-process test:

  bkfatigue10 63 fatigue lives (kcycles) -- used for the pedigree question only:
              is the BS law a good description of fatigue life?
  pipelinethickness  200 wall-thickness readings with no time index -- reported
              and set aside, because a single cross-section has no increments.

Analyses
  R1  Variance scaling.  For an additive process the variance of an m-step
      increment grows as m; interval scaling, fitted at one step and used at m,
      grows it as m^2.  Measured WITHIN each unit, so a unit-specific rate does
      not masquerade as increment variance.
  R2  Likelihood.  Where the two parameterisations are not observationally
      identical -- a state-dependent scale (alloya, Paris coupling) or an
      irregular grid (metalwear) -- both are fitted by maximum likelihood with
      the same number of parameters and compared unit by unit.
  R3  Pedigree.  BS against four other lifetime families on bkfatigue10.

    python code/study_realdata.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pyreadr
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gbs                                                        # noqa: E402

DATA = os.path.join(ROOT, "data_external", "smrd")
OUT = os.path.join(ROOT, "results", "realdata")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261006)
NBOOT = 4000


def load(name):
    return pyreadr.read_r(os.path.join(DATA, f"{name}.RData"))[name]


def paths(df, unit, t, x):
    return [(d.sort_values(t)[t].to_numpy(float), d.sort_values(t)[x].to_numpy(float))
            for _, d in df.groupby(unit)]


# ── R1: variance scaling within units ───────────────────────────────────────
def within_var(P, m, detrend=False):
    out = []
    for t, x in P:
        if detrend:
            #  remove the unit's own smooth trend so an accelerating mean does
            #  not count as increment variance: quadratic in time for the path
            #  by default; an integer gives the degree (sensitivity check)
            deg = 2 if detrend is True else int(detrend)
            c = np.polyfit(t, x, deg)
            x = x - np.polyval(c, t)
        inc = np.diff(x[::m])
        if len(inc) >= 3:
            out.append(np.var(inc, ddof=1))
    return float(np.mean(out))


def scaling(P, ms=(1, 2, 3, 4), detrend=False):
    ms = np.asarray(ms, float)
    V = np.array([within_var(P, int(m), detrend) for m in ms])
    b = float(np.polyfit(np.log(ms), np.log(V), 1)[0])
    boot = []
    for _ in range(NBOOT):
        Q = [P[i] for i in RNG.integers(0, len(P), len(P))]
        VV = np.array([within_var(Q, int(m), detrend) for m in ms])
        boot.append(np.polyfit(np.log(ms), np.log(VV), 1)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"m": ms.tolist(), "var": V.tolist(), "ratio": (V / V[0]).tolist(),
            "b": b, "b_lo": float(lo), "b_hi": float(hi),
            "units": len(P)}


# ── R2: likelihood of the two parameterisations ─────────────────────────────
def loglik(theta, inc, scale_fn, consistent):
    la, lb = theta[0], theta[1]
    alpha, beta = np.exp(la), np.exp(lb)
    s = scale_fn(theta[2:])
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        return -1e300
    if consistent:
        a_s, b_s = gbs.bs_consistent_params(alpha, beta, s)
    else:
        a_s, b_s = np.full_like(s, alpha), beta * s
    ll = gbs.bs_logpdf(inc, a_s, b_s)
    return float(np.sum(ll)) if np.all(np.isfinite(ll)) else -1e300


def fit(inc, scale_fn, x0, consistent):
    f = lambda th: -loglik(th, inc, scale_fn, consistent)
    best = None
    for jitter in (0.0, 0.3, -0.3, 0.6):
        r = optimize.minimize(f, np.asarray(x0) + jitter, method="Nelder-Mead",
                              options={"maxiter": 20000, "xatol": 1e-8,
                                       "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def merge_nonpositive(t, x):
    """Fold a non-positive increment (measurement noise) into the next one."""
    keep = [0]
    for i in range(1, len(x)):
        if x[i] - x[keep[-1]] > 0:
            keep.append(i)
    return t[keep], x[keep]


def r2_alloy(P, x_ref=1.0):
    rows = []
    for t, x in P:
        inc = np.diff(x)
        xprev = x[:-1]
        sfn = lambda th, xp=xprev: (1.0 + xp / x_ref) ** np.exp(th[0])
        m0 = np.mean(inc)
        x0 = [np.log(0.5), np.log(m0), np.log(1.0)]
        lc, thc = fit(inc, sfn, x0, True)
        ln, thn = fit(inc, sfn, x0, False)
        rows.append({"n": int(len(inc)), "ll_consistent": lc, "ll_interval": ln,
                     "gamma_consistent": float(np.exp(thc[2])),
                     "gamma_interval": float(np.exp(thn[2]))})
    return rows


def r2_wear(P):
    rows = []
    for t, x in P:
        t, x = merge_nonpositive(t, x)
        inc = np.diff(x)
        t0, t1 = t[:-1], t[1:]
        sfn = lambda th, a=t0, b=t1: b ** np.exp(th[0]) - a ** np.exp(th[0])
        x0 = [np.log(0.5), np.log(np.mean(inc) / np.mean(t1 ** 0.5 - t0 ** 0.5)),
              np.log(0.5)]
        lc, thc = fit(inc, sfn, x0, True)
        ln, thn = fit(inc, sfn, x0, False)
        rows.append({"n": int(len(inc)), "ll_consistent": lc, "ll_interval": ln,
                     "c_consistent": float(np.exp(thc[2])),
                     "c_interval": float(np.exp(thn[2]))})
    return rows


def summarise(rows):
    d = np.array([r["ll_consistent"] - r["ll_interval"] for r in rows])
    #  same number of parameters, so a log-likelihood difference is an AIC
    #  difference of twice that; the sign test asks how often it is positive
    k = int(np.sum(d > 0))
    p = float(stats.binomtest(k, len(d), 0.5, alternative="greater").pvalue)
    return {"units": len(d), "consistent_better": k, "sum_dll": float(d.sum()),
            "median_dll": float(np.median(d)), "sign_test_p": p}


# ── R3: pedigree on fatigue lives ───────────────────────────────────────────
def r3_pedigree(life):
    fams = {"Birnbaum--Saunders": stats.fatiguelife, "inverse Gaussian": stats.invgauss,
            "lognormal": stats.lognorm, "Weibull": stats.weibull_min,
            "gamma": stats.gamma}
    out = {}
    for name, d in fams.items():
        par = d.fit(life, floc=0)
        ll = float(np.sum(d.logpdf(life, *par)))
        k = len(par) - 1                                  # location fixed at 0
        out[name] = {"loglik": ll, "aic": 2 * k - 2 * ll}
    best = min(v["aic"] for v in out.values())
    for v in out.values():
        v["delta_aic"] = v["aic"] - best
    return out


if __name__ == "__main__":
    laser = paths(load("gaaslaser"), "unit", "hours", "increase")
    alloy = paths(load("alloya"), "specimen", "megacycles", "inches")
    wear = paths(load("metalwear"), "unit", "cycles", "microns")
    life = load("bkfatigue10")["kcycles"].to_numpy(float)
    pipe = load("pipelinethickness")

    res = {
        "R1_laser": scaling(laser),
        "R1_alloy_detrended": scaling(alloy, detrend=True),
        "R2_alloy": summarise(r2_alloy(alloy)),
        "R2_wear": summarise(r2_wear(wear)),
        "R3_fatigue_life": r3_pedigree(life),
        "pipeline": {"n": int(len(pipe)), "columns": list(pipe.columns),
                     "usable": False,
                     "reason": "single cross-section, no time index, no increments"},
    }
    res["R2_alloy_units"] = r2_alloy(alloy)
    res["R2_wear_units"] = r2_wear(wear)
    #  Sensitivity of the exponent to the detrending choice, run last so the
    #  headline numbers above keep their random stream.  Degree 0 is no
    #  detrending; a linear trend is removed by differencing anyway.
    res["R1_sensitivity"] = {
        name: {str(deg): {k: v for k, v in scaling(P, detrend=deg if deg else False).items()
                          if k in ("b", "b_lo", "b_hi")}
               for deg in (0, 2, 3)}
        for name, P in (("laser", laser), ("alloy", alloy))}
    json.dump(res, open(os.path.join(OUT, "realdata.json"), "w"), indent=2)

    for k in ("R1_laser", "R1_alloy_detrended"):
        r = res[k]
        print(f"{k:20s} b = {r['b']:.2f} [{r['b_lo']:.2f}, {r['b_hi']:.2f}]  "
              f"ratio {np.round(r['ratio'], 2)}  ({r['units']} units)")
    for k in ("R2_alloy", "R2_wear"):
        r = res[k]
        print(f"{k:20s} consistent better in {r['consistent_better']}/{r['units']}"
              f", sum dLL {r['sum_dll']:+.2f}, median {r['median_dll']:+.3f}, "
              f"sign-test p {r['sign_test_p']:.3g}")
    print("R3 fatigue lives  delta AIC:",
          {k: round(v['delta_aic'], 2) for k, v in res['R3_fatigue_life'].items()})
    print("pipeline: not usable --", res["pipeline"]["reason"])
