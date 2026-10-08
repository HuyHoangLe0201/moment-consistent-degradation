# -*- coding: utf-8 -*-
r"""End-to-end application of Algorithm 1 to Device-B, on a time clock.

Device-B (Meeker & Escobar; SMRD package, not redistributed): 34 devices at
150, 195 and 237 C, power drop read every 125 h until 4000, 2000 or 1000 h.
Degradation is the negated power drop; failure is a drop of 0.5 dB.  The paths
decelerate, so the process runs on the clock tau = t^p, with one p per
temperature estimated from the other devices at that temperature.

On the clock an increment over a step of clock length S has mean rho_i S, and

    consistent  Var = kappa rho_i^2 S        (squared CV kappa / S)
    interval    Var = c rho_i^2 S^2          (squared CV c, fixed shape).

The clock steps are unequal even on a uniform calendar grid, so the two
parameterisations differ on the record grid and each is fitted there in its
own form.  Everything else follows study_application.py: leave one device out,
records on one schedule and monitoring on the other, the 90 % interval for the
level at the end of test from origins at 1/4, 1/2 and 3/4 of it, and the
replacement rule at thresholds 0.4, 0.5 (failure) and 0.6 dB.

    python code/study_application_deviceb.py

Writes results/realdata/application_deviceb.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_application as SA                                   # noqa: E402
import study_realdata as SR                                      # noqa: E402
from study_application import crps, paired_ci, unit_ci           # noqa: E402

OUT = os.path.join(ROOT, "results", "realdata")
RNG = np.random.default_rng(int(os.environ.get("APP_SEED", 20261009)))
SA.RNG = RNG
NSIM = 20_000
ETA = 0.05
THRESHOLDS = (0.4, 0.5, 0.6)
DIRECTIONS = ((500.0, 125.0), (125.0, 500.0))
MODELS = ("consistent", "interval")


def load():
    B = SR.load("deviceb")
    B["deg"] = -B["powerdrop"]
    P = []
    for _, g in B.groupby("device"):
        g = g.sort_values("hours")
        P.append((int(g.celsius.iloc[0]), g.hours.to_numpy(float), g.deg.to_numpy(float)))
    return P


def thin(t, x, h):
    keep = np.isclose(np.mod(t, h), 0) | np.isclose(np.mod(t, h), h)
    return t[keep], x[keep]


def clock_p(P):
    """Common exponent of tau = t^p: within-unit regression of log x on log t."""
    lx, lt, u = [], [], []
    for i, (_, t, x) in enumerate(P):
        k = t > 0
        lx += list(np.log(x[k]))
        lt += list(np.log(t[k]))
        u += [i] * int(k.sum())
    lx, lt, u = map(np.asarray, (lx, lt, u))
    dm = lambda v: v - np.array([v[u == j].mean() for j in u])
    return float(np.sum(dm(lt) * dm(lx)) / np.sum(dm(lt) ** 2))


def fit(P, h, p):
    """kappa (consistent) and c (interval) from the record grid, with df."""
    num = {m: 0.0 for m in MODELS}
    df = 0
    for _, t, x in P:
        tt, xx = thin(t, x, h)
        S, y = np.diff(tt ** p), np.diff(xx)
        if len(y) < 2:
            continue
        rho = y.sum() / S.sum()
        e2 = (y - rho * S) ** 2 / rho ** 2
        num["consistent"] += np.sum(e2 / S)
        num["interval"] += np.sum(e2 / S ** 2)
        df += len(y) - 1
    return {m: num[m] / df for m in MODELS} | {"df": df}


def cv2_steps(model, k, S):
    return k / S if model == "consistent" else k * np.ones_like(S)


CLOCK_UNC = os.environ.get("APP_CLOCK", "1") == "1"


def unit_p(P):
    """Each unit's own clock exponent, from its path (log x on log t)."""
    out = []
    for _, t, x in P:
        k = t > 0
        out.append(float(np.polyfit(np.log(t[k]), np.log(x[k]), 1)[0]))
    return np.asarray(out)


def simulate(law, model, t_hist, x_hist, t_fut, p, k, df, p_units=None):
    """Paths of the readings at t_fut, from the last reading in the history.

    Uncertainties carried into every path: the variance parameter (scaled
    inverse chi-square on df), the unit's rate (its squared CV under the
    model's own variance claim) and, with CLOCK_UNC, the shape of the unit's
    clock -- p is drawn from the training units' own exponents, recentred on
    the common p, because units at one temperature curve differently."""
    kk = k * df / RNG.chisquare(df, (NSIM, 1))
    if CLOCK_UNC and p_units is not None:
        pp = p + (RNG.choice(p_units, (NSIM, 1)) - p_units.mean())
    else:
        pp = np.full((NSIM, 1), p)
    tau_h = t_hist[None, :] ** pp
    tau_k = tau_h[:, -1:]
    S_hist = np.diff(tau_h, axis=1)
    rho = x_hist[-1] / tau_k
    rel2 = np.sum(S_hist ** 2 * cv2_steps(model, kk, S_hist), axis=1,
                  keepdims=True) / tau_k ** 2
    r = rho * np.exp(np.sqrt(rel2) * RNG.standard_normal((NSIM, 1)) - rel2 / 2)
    S = np.diff(np.r_[t_hist[-1], t_fut][None, :] ** pp, axis=1)
    dx = SA.LAWS[law](1.0, cv2_steps(model, kk, S), S.shape) * r * S
    return x_hist[-1] + np.cumsum(dx, axis=1)


def run(law, h_ref, h_dep):
    P = load()
    lev = {m: [] for m in MODELS}
    pol = {m: [] for m in MODELS}
    ps = []
    for i, (temp, t, x) in enumerate(P):
        train = [q for j, q in enumerate(P) if j != i and q[0] == temp]
        p = clock_p(train)
        F = fit(train, h_ref, p)
        pu = unit_p(train)
        ps.append(p)
        tt, xx = thin(t, x, h_dep)
        T = tt[-1]
        for frac in (0.25, 0.5, 0.75):
            k = int(np.argmin(np.abs(tt - frac * T)))
            if k < 1 or k >= len(tt) - 1:
                continue
            obs = xx[-1]
            for m in MODELS:
                S = simulate(law, m, tt[:k + 1], xx[:k + 1], tt[k + 1:], p, F[m], F["df"], pu)
                end = S[:, -1]
                lo8, hi8, lo, hi = np.quantile(end, [0.1, 0.9, 0.05, 0.95])
                lev[m].append({"unit": i, "temp": temp, "origin": float(tt[k]),
                               "obs": float(obs),
                               "in80": bool(lo8 <= obs <= hi8),
                               "in90": bool(lo <= obs <= hi),
                               "w90": float(hi - lo), "crps": crps(end, obs),
                               "is90": float(hi - lo + 20 * max(lo - obs, 0)
                                             + 20 * max(obs - hi, 0))})
        for D in THRESHOLDS:
            hit = np.nonzero(xx >= D)[0]
            if not len(hit):
                continue
            T_obs = float(tt[hit[0]])
            for m in MODELS:
                t_rep = np.inf
                for j in range(1, len(tt)):
                    if tt[j] >= T_obs:
                        break
                    S = simulate(law, m, tt[:j + 1], xx[:j + 1], tt[j + 1:j + 2], p,
                                 F[m], F["df"], pu)
                    if np.mean(S[:, 0] >= D) > ETA:
                        t_rep = float(tt[j])
                        break
                pol[m].append({"unit": i, "D": D, "T_obs": T_obs, "t_rep": t_rep,
                               "missed": bool(t_rep > T_obs)})
    res = {"law": law, "h_ref": h_ref, "h_dep": h_dep, "p_mean": float(np.mean(ps)),
           "crps_diff": paired_ci(lev["consistent"], lev["interval"], "crps"),
           "is90_diff": paired_ci(lev["consistent"], lev["interval"], "is90")}
    for m in MODELS:
        L, Q = lev[m], pol[m]
        res[m] = {
            "cover80": float(np.mean([r["in80"] for r in L])),
            "cover90": float(np.mean([r["in90"] for r in L])),
            "ci80": unit_ci(L, "in80"), "ci90": unit_ci(L, "in90"),
            "width90": float(np.mean([r["w90"] for r in L])),
            "crps": float(np.mean([r["crps"] for r in L])),
            "n_level": len(L), "n_events": len(Q),
            "missed": int(sum(r["missed"] for r in Q)),
            "mean_lead": float(np.mean([r["T_obs"] - r["t_rep"] for r in Q
                                        if not r["missed"]])),
            "levels": L, "policy": Q,
        }
    return res


def clock_exponent(P, ms=(1, 2, 4)):
    """Growth exponent b of the within-unit variance of m-step increments on the
    clock, Var ~ S^b: 1 for independent increments, above 1 when they are
    positively correlated.  One clock exponent p per temperature."""
    ptemp = {T: clock_p([q for q in P if q[0] == T]) for T in {q[0] for q in P}}
    V, Sm = [], []
    for m in ms:
        num, df, Ss = 0.0, 0, []
        for temp, t, x in P:
            tau = t ** ptemp[temp]
            S, y = np.diff(tau[::m]), np.diff(x[::m])
            if len(y) < 2:
                continue
            rho = y.sum() / S.sum()
            num += np.sum((y - rho * S) ** 2) / rho ** 2
            df += len(y) - 1
            Ss += list(S)
        V.append(num / df)
        Sm.append(np.mean(Ss))
    return float(np.polyfit(np.log(Sm), np.log(V), 1)[0])


def exponents():
    P = load()
    rng = np.random.default_rng(20261009)
    out = {}
    for key, sub in (("all", P), ("150C", [q for q in P if q[0] == 150])):
        b = clock_exponent(sub)
        bs = [clock_exponent([sub[j] for j in rng.integers(0, len(sub), len(sub))])
              for _ in range(2000)]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        out[key] = {"b": b, "b_lo": float(lo), "b_hi": float(hi), "units": len(sub)}
        print(f"clock exponent {key:5s} b = {b:.2f} [{lo:.2f}, {hi:.2f}] ({len(sub)} units)")
    with open(os.path.join(OUT, "deviceb_clock.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    if os.environ.get("APP_EXP", "1") == "1":
        exponents()
    out = []
    for law in SA.LAWS:
        for h_ref, h_dep in DIRECTIONS:
            r = run(law, h_ref, h_dep)
            out.append(r)
            print(f"{law:8s} {h_ref:.0f} -> {h_dep:.0f}  p {r['p_mean']:.2f}"
                  f"  CRPS {np.round(r['crps_diff'], 4)}  IS90 {np.round(r['is90_diff'], 3)}")
            for m in MODELS:
                s = r[m]
                print(f"   {m:10s} cov80 {s['cover80']:.2f} {np.round(s['ci80'], 2)}"
                      f" cov90 {s['cover90']:.2f} {np.round(s['ci90'], 2)} (n {s['n_level']})"
                      f" w90 {s['width90']:.3f} missed {s['missed']}/{s['n_events']}"
                      f" lead {s['mean_lead']:.0f}")
    with open(os.path.join(OUT, "application_deviceb.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
