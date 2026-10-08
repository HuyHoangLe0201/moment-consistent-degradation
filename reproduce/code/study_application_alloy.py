# -*- coding: utf-8 -*-
r"""End-to-end application of Algorithm 1 to the Alloy-A cracks, with coupling.

Same protocol as study_application.py (the lasers), with the damage coupling
switched on.  The step at state x over an interval h has the dimensionless size

    s = (h / h_ref) * q(x)^gamma,       q(x) = 1 + x / x_ref,   x_ref = 1 inch,

mean m_i * s for unit i, and squared CV
    consistent  cv2_ref / s
    interval    cv2_ref            (fixed shape).
So the two parameterisations differ even on the record grid, and each is fitted
there in its own form.

Protocol (leave one specimen out, 21 folds)
  fit      gamma by least squares of log increment on log s, with a level per
           unit; cv2_ref per model from the normalised increments;
  deploy   the held-out specimen is read every h_dep.  At an origin its level
           m_i comes from its own readings, with the uncertainty that the
           model itself implies carried into every path.
  check    (a) the 90 % interval for the crack length 0.04 Mcycles ahead;
           (b) the replacement rule of Section 8 at thresholds 1.3 ... 1.6 in.

Records every 0.02 Mcycles and monitoring every 0.01, and the reverse.  The
specimens are short (at most 12 readings), so a factor of two is the largest
change that leaves enough records.  Readings are rounded to 0.01 inch.

    python code/study_application_alloy.py

Writes results/realdata/application_alloy.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_realdata as SR                                       # noqa: E402
from study_weibull_check import nu_of                             # noqa: E402
from study_application import crps, paired_ci, unit_ci            # noqa: E402

OUT = os.path.join(ROOT, "results", "realdata")
RNG = np.random.default_rng(int(os.environ.get("APP_SEED", 20261008)))
NSIM = 20_000
X_REF = 1.0
ETA = 0.05
H_AHEAD = 0.04
ORIGINS = (0.02, 0.04, 0.06)
THRESHOLDS = (1.3, 1.4, 1.5, 1.6)                  # inches; 1.6 is failure
DIRECTIONS = ((0.02, 0.01), (0.01, 0.02))          # (h_ref, h_dep)
MODELS = ("consistent", "interval")
RES = 0.01                                         # readings rounded to 0.01 in
SIG2_E = RES ** 2 / 12                             # uniform rounding error
ROUND = int(os.environ.get("APP_ROUND", "1"))      # 1: Sheppard (main), 2: free error
PARAM_UNC = os.environ.get("APP_PARAM", "1") == "1"
#  1 (main): every path draws its own gamma from a bootstrap over the training
#  specimens, so the uncertainty of the coupling exponent enters every
#  prediction; 0: the point estimate, as in the first version of the paper
GAMMA_UNC = os.environ.get("APP_GAMMA_UNC", "1") == "1"
NB_GAMMA = 1000
NBOOT = 4000


def q(x):
    return 1.0 + np.asarray(x) / X_REF


def bs_steps(mean, cv2):
    r = np.minimum(cv2, 4.99)                       # Algorithm 1: r < 5
    u = np.sqrt(1 + 3 * r)
    A = 2 * r * (u + 4) / ((5 - r) * (u + 1))
    a = np.sqrt(A)
    z = RNG.standard_normal(np.shape(mean))
    return mean / (1 + A / 2) * (a * z / 2 + np.sqrt((a * z / 2) ** 2 + 1)) ** 2


def weib_steps(mean, cv2):
    nu = nu_of(np.broadcast_to(cv2, np.shape(mean)))
    lam = mean / special.gamma(1 + 1 / nu)
    return lam * (-np.log(RNG.random(np.shape(mean)))) ** (1 / nu)


def gamma_steps(mean, cv2):
    k = 1.0 / np.broadcast_to(cv2, np.shape(mean))
    return RNG.gamma(k, mean / k)


def ig_steps(mean, cv2):
    return RNG.wald(mean, mean / np.broadcast_to(cv2, np.shape(mean)))


LAWS = {"BS": bs_steps, "Weibull": weib_steps, "gamma": gamma_steps, "IG": ig_steps}


def thin(t, x, h):
    k = np.round(t / h, 6)
    keep = np.isclose(k, np.round(k))
    return t[keep], x[keep]


def cv2_of(model, cv2_ref, s):
    return cv2_ref / s if model == "consistent" else cv2_ref * np.ones_like(s)


def meas_var(P, h_ref, gamma, model):
    """Variance of the reading error.  ROUND = 0: none; 1: rounding only
    (Sheppard, RES^2/12); 2: estimated.  Blocks of m record steps have
    clock length S and, around the unit's trend, variance 2*sig2 + c*S^p with
    p = 1 (consistent) or 2 (interval); the intercept over m = 1, 2, 3 gives
    sig2, floored at the rounding variance."""
    if ROUND < 2:
        return ROUND * SIG2_E
    V, S = [], []
    for m in (1, 2, 3):
        num, df, Ss = 0.0, 0, []
        for t, x in P:
            tt, xx = thin(t, x, h_ref)
            sj, yj = q(xx[:-1]) ** gamma, np.diff(xx)
            nb = len(yj) // m
            if nb < 2:
                continue
            Sb = sj[:nb * m].reshape(nb, m).sum(1)
            yb = yj[:nb * m].reshape(nb, m).sum(1)
            mu = yb.sum() / Sb.sum()
            num += np.sum((yb - mu * Sb) ** 2)
            df += nb - 1
            Ss += list(Sb)
        V.append(num / df)
        S.append(np.mean(Ss))
    p = 1 if model == "consistent" else 2
    c = np.polyfit(np.array(S) ** p, V, 1)
    return float(max(c[1] / 2, SIG2_E))


def fit(P, h_ref):
    """gamma (shared by both models) and cv2_ref for each model."""
    ys, ss, ids = [], [], []
    for i, (t, x) in enumerate(P):
        tt, xx = thin(t, x, h_ref)
        ys.append(np.diff(xx))
        ss.append(np.log(q(xx[:-1])))
        ids.append(np.full(len(xx) - 1, i))
    y, lq, u = np.concatenate(ys), np.concatenate(ss), np.concatenate(ids)
    ly = np.log(y)
    # within-unit regression: demean by unit
    dm = lambda v: v - np.array([v[u == k].mean() for k in u])
    gamma = float(np.sum(dm(lq) * dm(ly)) / np.sum(dm(lq) ** 2))
    s = np.exp(gamma * lq)                          # q^gamma; h = h_ref here
    # the same estimator on specimens resampled with replacement: per-unit
    # sums of the demeaned cross products and squares are what it adds up
    dlq, dly = dm(lq), dm(ly)
    units = np.unique(u)
    sxy = np.array([np.sum(dlq[u == k] * dly[u == k]) for k in units])
    sxx = np.array([np.sum(dlq[u == k] ** 2) for k in units])
    pick = RNG.integers(0, len(units), (NB_GAMMA, len(units)))
    out = {"gamma": gamma,
           "gamma_boot": sxy[pick].sum(1) / sxx[pick].sum(1)}
    for model in MODELS:
        sig2 = meas_var(P, h_ref, gamma, model)
        out[model + "_sig2"] = sig2
        num, df = 0.0, 0
        for k in np.unique(u):
            sk, yk = s[u == k], y[u == k]
            mk = yk.sum() / sk.sum()
            z = yk / (mk * sk) - 1
            w = sk if model == "consistent" else np.ones_like(sk)
            # an observed increment carries the measurement error of both
            # readings, variance 2*sig2, which is not process variance
            num += np.sum(w * (z ** 2 - 2 * sig2 / (mk * sk) ** 2))
            df += len(z) - 1
        out[model] = max(num / df, 1e-6)
    out["df"] = df
    return out


def simulate(law, model, x0, tt_hist, xx_hist, h, h_ref, gamma, cv2_ref, n,
             df=None, sig2=0.0, gamma_boot=None):
    """Observed readings for n steps after the last reading of a unit.

    Carried uncertainties: the unit's level m_i (variance implied by the model,
    plus the rounding of the two end readings), cv2_ref (scaled inverse
    chi-square on df), the coupling exponent gamma (one bootstrap draw per
    path, when GAMMA_UNC), the true current length (reading minus a uniform
    rounding error) and the rounding of the future reading."""
    if df is not None and PARAM_UNC:
        cv2_ref = cv2_ref * df / RNG.chisquare(df, NSIM)
    if gamma_boot is not None and GAMMA_UNC:
        gamma = RNG.choice(gamma_boot, NSIM)
    g = np.atleast_1d(gamma)[:, None]                 # one exponent per path
    s_hist = (np.diff(tt_hist) / h_ref)[None, :] * q(xx_hist[:-1])[None, :] ** g
    y_hist = np.diff(xx_hist)
    m = y_hist.sum() / s_hist.sum(1)
    cv2h = cv2_of(model, np.atleast_1d(cv2_ref)[:, None], s_hist)
    rel2 = (np.sum(s_hist ** 2 * cv2h, axis=1) / s_hist.sum(1) ** 2
            + 2 * sig2 / y_hist.sum() ** 2)
    sd = np.sqrt(rel2)
    mi = m * np.exp(sd * RNG.standard_normal(NSIM) - sd ** 2 / 2)
    X = np.full(NSIM, float(x0))
    if ROUND == 1:
        X = X - RNG.uniform(-RES / 2, RES / 2, NSIM)
    elif ROUND == 2:
        X = X - np.sqrt(sig2) * RNG.standard_normal(NSIM)
    extra = np.sqrt(max(sig2 - SIG2_E, 0.0)) if ROUND == 2 else 0.0
    out = np.empty((NSIM, n))
    for j in range(n):
        s = (h / h_ref) * q(X) ** gamma
        X = X + LAWS[law](mi * s, cv2_of(model, cv2_ref, s))
        obs = X + extra * RNG.standard_normal(NSIM) if extra else X
        out[:, j] = np.round(obs / RES) * RES if ROUND else obs
    return out


def first_cross(tt, xx, D):
    hit = np.nonzero(xx >= D)[0]
    return float(tt[hit[0]]) if len(hit) else np.inf


def run(law, h_ref, h_dep):
    P = SR.paths(SR.load("alloya"), "specimen", "megacycles", "inches")
    lev = {m: [] for m in MODELS}
    pol = {m: [] for m in MODELS}
    fits = []
    for i, (t, x) in enumerate(P):
        F = fit([p for j, p in enumerate(P) if j != i], h_ref)
        fits.append(F)
        tt, xx = thin(t, x, h_dep)
        nH = int(round(H_AHEAD / h_dep))
        for tk in ORIGINS:
            k = int(np.argmin(np.abs(tt - tk)))
            if not np.isclose(tt[k], tk) or k + nH >= len(tt):
                continue
            obs = xx[k + nH]
            for m in MODELS:
                S = simulate(law, m, xx[k], tt[:k + 1], xx[:k + 1], h_dep, h_ref,
                             F["gamma"], F[m], nH, F["df"], F[m + "_sig2"],
                             F["gamma_boot"])[:, -1]
                lo8, hi8, lo, hi = np.quantile(S, [0.1, 0.9, 0.05, 0.95])
                lev[m].append({"unit": i, "origin": tk, "obs": float(obs),
                               "in80": bool(lo8 <= obs <= hi8),
                               "in90": bool(lo <= obs <= hi),
                               "w90": float(hi - lo),
                               "crps": crps(S, obs),
                               "is90": float(hi - lo + 20 * max(lo - obs, 0)
                                             + 20 * max(obs - hi, 0))})
        for D in THRESHOLDS:
            T_obs = first_cross(tt, xx, D)
            if not np.isfinite(T_obs):
                continue
            for m in MODELS:
                t_rep = np.inf
                for j in range(1, len(tt)):
                    if tt[j] >= T_obs - 1e-9:
                        break
                    if tt[j] < 0.02 - 1e-9:
                        continue
                    S = simulate(law, m, xx[j], tt[:j + 1], xx[:j + 1], h_dep,
                                 h_ref, F["gamma"], F[m], 1, F["df"], F[m + "_sig2"],
                                 F["gamma_boot"])
                    if np.mean(S[:, 0] >= D - 1e-9) > ETA:
                        t_rep = float(tt[j])
                        break
                pol[m].append({"unit": i, "D": D, "T_obs": T_obs, "t_rep": t_rep,
                               "missed": bool(t_rep > T_obs)})
    res = {"law": law, "h_ref": h_ref, "h_dep": h_dep,
           "gamma_mean": float(np.mean([f["gamma"] for f in fits])),
           "gamma_boot_sd": float(np.mean([np.std(f["gamma_boot"]) for f in fits])),
           "gamma_unc": GAMMA_UNC,
           "sigma_meas": {m: float(np.sqrt(np.mean([f[m + "_sig2"] for f in fits])))
                          for m in MODELS},
           "cv2_ref_mean": {m: float(np.mean([f[m] for f in fits])) for m in MODELS},
           "crps_diff": paired_ci(lev["consistent"], lev["interval"], "crps"),
           "is90_diff": paired_ci(lev["consistent"], lev["interval"], "is90")}
    for m in MODELS:
        L, Q = lev[m], pol[m]
        res[m] = {
            "cover80": float(np.mean([r["in80"] for r in L])),
            "cover90": float(np.mean([r["in90"] for r in L])),
            "ci80": unit_ci(L, "in80"), "ci90": unit_ci(L, "in90"),
            "crps": float(np.mean([r["crps"] for r in L])),
            "is90": float(np.mean([r["is90"] for r in L])),
            "width90": float(np.mean([r["w90"] for r in L])),
            "n_level": len(L),
            "n_events": len(Q),
            "missed": int(sum(r["missed"] for r in Q)),
            "missed_D16": int(sum(r["missed"] for r in Q if r["D"] == 1.6)),
            "n_events_D16": int(sum(r["D"] == 1.6 for r in Q)),
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
            print(f"{law:8s} ref {h_ref:.2f} -> dep {h_dep:.2f}  gamma "
                  f"{r['gamma_mean']:.2f}  cv2_ref {r['cv2_ref_mean']}"
                  f"  sd_meas {r['sigma_meas']}  CRPS {np.round(r['crps_diff'], 4)}  IS90 {np.round(r['is90_diff'], 3)}")
            for m in MODELS:
                s = r[m]
                print(f"   {m:10s} cov80 {s['cover80']:.2f} {np.round(s['ci80'], 2)} cov90 {s['cover90']:.2f} (n {s['n_level']})"
                      f" w90 {s['width90']:.3f}  missed {s['missed']}/{s['n_events']}"
                      f" (1.6: {s['missed_D16']}/{s['n_events_D16']})"
                      f" lead {s['mean_lead']:.4f}")
    with open(os.path.join(OUT, "application_alloy.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
