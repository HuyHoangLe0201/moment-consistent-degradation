# -*- coding: utf-8 -*-
r"""Analyses added in response to the review of paper 1 (short version).

A  The process behind the construction.  A BS random walk on a base grid of
   step s0 is a genuine stochastic process.  An increment over k base steps is
   the k-fold convolution of BS(alpha_s0, beta_s0); Theorem 1 replaces it by the
   BS law with the same two moments.  We measure the supremum distance between
   the two laws, and the same distance for interval scaling, by numerical
   convolution.  The skewness of the exact sum is gamma0/sqrt(k).

B  The third cumulant of X(T) across grids, in closed form, for the grid study
   of Table D.1 (alpha = 0.4, beta = 1, T = 20).

C  Which increment family do the data prefer?  By Proposition 2 the
   moment-consistent version of every scale family follows the same schedule:
   a step of relative length s has mean m s and squared coefficient of
   variation c2/s.  Families differ only in the shape of the law, so they can be
   compared with equal numbers of parameters.  BS, gamma, inverse Gaussian,
   lognormal and Weibull are fitted unit by unit by maximum likelihood to the
   laser (fixed grid), metal-wear (irregular grid, s_k = t_k^c - t_{k-1}^c) and
   Alloy-A (Paris-coupled, s_k = q(x_{k-1})^gamma) increments.

D  Rate at which the IG-approximation error decays, fitted on Table D.4.

    python code/study_review.py      ->  results/theory/review_extras.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gbs                                                        # noqa: E402
import study_realdata as SR                                       # noqa: E402

OUT = os.path.join(ROOT, "results", "theory", "review_extras.json")


# ── A: base walk against the moment-matched BS law ───────────────────────────
def bs_skew(a):
    return 4 * a * (11 * a * a + 6) / (5 * a * a + 4) ** 1.5


def sum_law_distance(a_ref, k, s0):
    """sup |F_sum - F_BS(moment-matched)| and the same for interval scaling."""
    a0, b0 = gbs.bs_consistent_params(a_ref, 1.0, s0)
    a0, b0 = float(a0), float(b0)
    m0, v0 = float(gbs.bs_mean(a0, b0)), float(gbs.bs_var(a0, b0))
    hi = k * m0 + 14 * np.sqrt(k * v0) + 40 * m0
    dx = min(m0, np.sqrt(v0)) / 400
    x = np.arange(dx / 2, hi, dx)
    p = gbs.bs_pdf(x, a0, b0) * dx
    p /= p.sum()
    L = 1 << int(np.ceil(np.log2(len(x) * 2)))
    f = np.fft.irfft(np.fft.rfft(p, L) ** k, L)[: len(x)]
    F = np.cumsum(np.clip(f, 0, None))
    xs = x + (k - 1) * dx / 2 + dx / 2           # right edge of each summed cell
    a_s, b_s = gbs.bs_consistent_params(a_ref, 1.0, k * s0)
    d_cons = float(np.max(np.abs(F - gbs.bs_cdf(xs, float(a_s), float(b_s)))))
    #  interval scaling from the same base step: BS(alpha_s0, k beta_s0)
    d_int = float(np.max(np.abs(F - gbs.bs_cdf(xs, a0, k * b0))))
    return d_cons, d_int, float(bs_skew(a0) / np.sqrt(k)), float(bs_skew(float(a_s)))


def weibull_sum_distance(cv2_ref, k, s0, m_ref=1.0):
    """Same as sum_law_distance for a Weibull random walk with the given
    squared CV per unit time.  The base cell masses come from CDF differences,
    because a Weibull density with shape below one is unbounded at zero."""
    def law(mean, cv2):
        sh = _weibull_k(cv2)
        return stats.weibull_min(sh, scale=mean / special.gamma(1 + 1 / sh))
    base = law(m_ref * s0, cv2_ref / s0)
    m0, v0 = base.mean(), base.var()
    hi = k * m0 + 14 * np.sqrt(k * v0) + 40 * m0
    dx = min(m0, np.sqrt(v0)) / 400
    edges = np.arange(0.0, hi + dx, dx)
    p = np.diff(base.cdf(edges))
    p /= p.sum()
    L = 1 << int(np.ceil(np.log2(len(p) * 2)))
    f = np.fft.irfft(np.fft.rfft(p, L) ** k, L)[: len(p)]
    F = np.cumsum(np.clip(f, 0, None))
    xs = edges[1:] + (k - 1) * dx / 2 + dx / 2
    cons = law(k * m_ref * s0, cv2_ref / (k * s0))
    sh0 = _weibull_k(cv2_ref / s0)
    interval = stats.weibull_min(sh0, scale=k * base.kwds["scale"] if hasattr(base, "kwds") and "scale" in base.kwds
                                 else k * m_ref * s0 / special.gamma(1 + 1 / sh0))
    return (float(np.max(np.abs(F - cons.cdf(xs)))),
            float(np.max(np.abs(F - interval.cdf(xs)))))


def part_a_weibull():
    g = lambda a: a * a * (1 + 1.25 * a * a) / (1 + 0.5 * a * a) ** 2
    out = []
    for a_ref in (0.2, 0.4, 0.8):
        cv2 = g(a_ref)                     # the same CV^2 as the BS rows
        for k in (2, 4, 8, 16, 64):
            dc, di = weibull_sum_distance(cv2, k, 1 / 8)
            out.append({"cv2_ref": cv2, "alpha_equiv": a_ref, "s0": 1 / 8, "k": k,
                        "sup_consistent": dc, "sup_interval": di})
            print(f"AW cv2={cv2:.3f} k={k:3d}  sup cons {dc:.4f}  sup interval {di:.3f}")
    return out


def part_a():
    out = []
    g = lambda a: a * a * (1 + 1.25 * a * a) / (1 + 0.5 * a * a) ** 2
    for a_ref in (0.2, 0.4, 0.8):
        #  the base grid must itself be admissible, r = g/s0 < 5 (Remark 2):
        #  alpha = 0.8 admits at most 7.6 steps per unit, so its base step is 1/4
        for s0 in ((1 / 8,) if g(a_ref) * 8 < 5 else (1 / 4,)):
            for k in (2, 4, 8, 16, 64):
                dc, di, sk_sum, sk_bs = sum_law_distance(a_ref, k, s0)
                out.append({"alpha_ref": a_ref, "s0": s0, "k": k, "sup_consistent": dc,
                            "sup_interval": di, "skew_sum": sk_sum, "skew_bs": sk_bs})
                print(f"A alpha={a_ref} k={k:3d}  sup cons {dc:.4f}  sup interval {di:.3f}"
                      f"  skew sum {sk_sum:.3f} BS {sk_bs:.3f}")
    return out


# ── B: third cumulant of X(T) across grids ───────────────────────────────────
def part_b():
    T, a, b = 20.0, 0.4, 1.0
    rows = []
    for n in (1, 2, 4, 8, 16, 32, 64):
        s = T / n
        a_s, b_s = (float(v) for v in gbs.bs_consistent_params(a, b, s))
        k3 = n * 0.5 * b_s ** 3 * a_s ** 4 * (6 + 11 * a_s ** 2)
        V = n * float(gbs.bs_var(a_s, b_s))
        rows.append({"n": n, "kappa3": k3, "skewness": k3 / V ** 1.5})
    return rows


# ── C: increment family, all in moment-consistent form ───────────────────────
def _weibull_k(cv2):
    g = lambda k: special.gamma(1 + 2 / k) / special.gamma(1 + 1 / k) ** 2 - 1 - cv2
    return optimize.brentq(g, 0.02, 500.0)


def logpdf(fam, x, mean, cv2):
    """Log density of a law with given mean and squared CV, per family."""
    if fam == "BS":
        if np.any(cv2 >= 5):
            return -np.inf * np.ones_like(x)
        u = np.sqrt(1 + 3 * cv2)
        A = 2 * cv2 * (u + 4) / ((5 - cv2) * (u + 1))
        return gbs.bs_logpdf(x, np.sqrt(A), mean / (1 + A / 2))
    if fam == "gamma":
        k = 1 / cv2
        return stats.gamma.logpdf(x, k, scale=mean / k)
    if fam == "inverse Gaussian":
        eta = mean / cv2
        return stats.invgauss.logpdf(x, mean / eta, scale=eta)
    if fam == "lognormal":
        s2 = np.log1p(cv2)
        return stats.norm.logpdf(np.log(x), np.log(mean) - s2 / 2, np.sqrt(s2)) - np.log(x)
    if fam == "Weibull":
        cv2 = np.broadcast_to(cv2, x.shape)
        ks = np.array([_weibull_k(c) for c in cv2])
        lam = mean / special.gamma(1 + 1 / ks)
        return stats.weibull_min.logpdf(x, ks, scale=lam)
    raise ValueError(fam)


FAMS = ("BS", "gamma", "inverse Gaussian", "lognormal", "Weibull")


def cdf(fam, x, mean, cv2):
    """Distribution function of a law with given mean and squared CV."""
    if fam == "BS":
        if np.any(cv2 >= 5):
            return np.full_like(x, np.nan)
        u = np.sqrt(1 + 3 * cv2)
        A = 2 * cv2 * (u + 4) / ((5 - cv2) * (u + 1))
        return gbs.bs_cdf(x, np.sqrt(A), mean / (1 + A / 2))
    if fam == "gamma":
        return stats.gamma.cdf(x, 1 / cv2, scale=mean * cv2)
    if fam == "inverse Gaussian":
        eta = mean / cv2
        return stats.invgauss.cdf(x, mean / eta, scale=eta)
    if fam == "lognormal":
        s2 = np.log1p(cv2)
        return stats.norm.cdf((np.log(np.maximum(x, 1e-300)) - np.log(mean) + s2 / 2) / np.sqrt(s2))
    if fam == "Weibull":
        cv2 = np.broadcast_to(cv2, x.shape)
        ks = np.array([_weibull_k(c) for c in cv2])
        return stats.weibull_min.cdf(x, ks, scale=mean / special.gamma(1 + 1 / ks))
    raise ValueError(fam)


def fit_unit(fam, inc, scale_fn, nextra, h=None):
    """Maximise over log m, log c2 and the extra scale parameters.

    With h, each increment is known only to lie within +-h of its recorded
    value (both ends of a difference were rounded), and the likelihood is the
    probability of that interval instead of a density.
    """
    def nll(th):
        s = scale_fn(th[2:]) if nextra else np.ones_like(inc)
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            return 1e300
        with np.errstate(all="ignore"):
            try:
                if h is None:
                    ll = logpdf(fam, inc, np.exp(th[0]) * s, np.exp(th[1]) / s)
                else:
                    m_, c_ = np.exp(th[0]) * s, np.exp(th[1]) / s
                    pr = cdf(fam, inc + h, m_, c_) - cdf(fam, np.maximum(inc - h, 0.0), m_, c_)
                    ll = np.log(pr)
            except ValueError:
                return 1e300
        return 1e300 if not np.all(np.isfinite(ll)) else -float(np.sum(ll))
    m0 = np.mean(inc / (scale_fn([0.0] * nextra) if nextra else 1.0))
    best = None
    for c0 in (0.05, 0.3, 1.0):
        for e0 in ((-0.7, 0.0, 0.5) if nextra else (None,)):
            x0 = [np.log(m0), np.log(c0)] + ([e0] if nextra else [])
            r = optimize.minimize(nll, x0, method="Nelder-Mead",
                                  options={"maxiter": 8000, "xatol": 1e-7, "fatol": 1e-9})
            if best is None or r.fun < best.fun:
                best = r
    return -best.fun, best.x


def part_c():
    laser = SR.paths(SR.load("gaaslaser"), "unit", "hours", "increase")
    alloy = SR.paths(SR.load("alloya"), "specimen", "megacycles", "inches")
    wear = SR.paths(SR.load("metalwear"), "unit", "cycles", "microns")
    sets = {}
    #  laser: fixed 250-hour grid, s = 1 for every step
    sets["laser"] = [(np.diff(SR.merge_nonpositive(t, x)[1]), None, 0) for t, x in laser]
    #  wear: irregular grid, power time scale
    rows = []
    for t, x in wear:
        t, x = SR.merge_nonpositive(t, x)
        t0, t1 = t[:-1], t[1:]
        rows.append((np.diff(x), (lambda th, a=t0, b=t1: (b ** np.exp(th[0]) - a ** np.exp(th[0]))
                                  / (b[-1] ** np.exp(th[0]) / len(b))), 1))
    sets["metal wear"] = rows
    #  Alloy-A: Paris-coupled scale, x_ref = 1 inch as in R2
    rows = []
    for t, x in alloy:
        t, x = SR.merge_nonpositive(t, x)
        xp = x[:-1]
        rows.append((np.diff(x), (lambda th, xp=xp: (1.0 + xp) ** np.exp(th[0])), 1))
    sets["Alloy-A"] = rows
    #  Robustness.  Crack lengths are rounded to 0.01 inch, the size of the
    #  smallest increments, so a density likelihood may favour the wrong shape:
    #  refit with the probability of the rounding interval.  For wear, drop the
    #  units whose raw record has a negative step instead of merging it.
    sets["Alloy-A, rounding-aware"] = [(i, f, n, 0.01) for i, f, n in rows]
    keep = [k for k, (t, x) in enumerate(wear) if np.all(np.diff(x) > 0)]
    sets["metal wear, no merged steps"] = [sets["metal wear"][k] for k in keep]
    res = {}
    rng = np.random.default_rng(31)
    for name, units in sets.items():
        ll = {f: 0.0 for f in FAMS}
        wins = {f: 0 for f in FAMS}
        npar = 0
        cv2s = []
        per_unit = []
        for unit in units:
            inc, sfn, ne = unit[:3]
            h = unit[3] if len(unit) > 3 else None
            per = {}
            for f in FAMS:
                l, th = fit_unit(f, inc, sfn, ne, h)
                per[f] = l
                ll[f] += l
                if f == "BS":
                    cv2s.append(float(np.exp(th[1])))
            per_unit.append(per)
            wins[max(per, key=per.get)] += 1
            npar += 2 + ne
        best = max(ll.values())
        #  uncertainty: resample units; and a paired sign test, Weibull vs BS
        L = np.array([[p[f] for f in FAMS] for p in per_unit])
        boot = []
        for _ in range(4000):
            tot = L[rng.integers(0, len(L), len(L))].sum(axis=0)
            boot.append(tot)
        boot = np.array(boot)
        iw, ib = FAMS.index("Weibull"), FAMS.index("BS")
        d_bw = 2 * (boot[:, iw] - boot[:, ib])           # Delta AIC, BS minus Weibull
        k_w = int(np.sum(L[:, iw] > L[:, ib]))
        res[name] = {"units": len(units), "parameters": npar,
                     "loglik": ll, "delta_aic": {f: 2 * (best - v) for f, v in ll.items()},
                     "units_best": wins, "median_step_cv2_BS": float(np.median(cv2s)),
                     "per_unit": per_unit,
                     "boot_dAIC_BS_minus_Weibull": [float(np.percentile(d_bw, q)) for q in (2.5, 50, 97.5)],
                     "boot_P_weibull_best": float(np.mean(np.argmax(boot, axis=1) == iw)),
                     "weibull_beats_BS": k_w,
                     "sign_test_p": float(stats.binomtest(k_w, len(L), 0.5).pvalue)}
        print(f"C {name:28s}", {f: round(2 * (best - v), 1) for f, v in ll.items()}, wins,
              "boot dAIC(BS-W)", np.round(res[name]["boot_dAIC_BS_minus_Weibull"], 1),
              "P(W best)", round(res[name]["boot_P_weibull_best"], 2),
              f"W>BS {k_w}/{len(L)} p={res[name]['sign_test_p']:.3f}")
    return res


# ── D: decay rate of the IG-approximation error ───────────────────────────────
def part_d():
    d = json.load(open(os.path.join(ROOT, "results", "theory", "theory_results.json"),
                       encoding="utf-8"))["T2"]
    nd = np.log(np.array(d["n_d"], float))
    out = {}
    for a, c, l in zip(d["alphas"], d["sup_continuous"], d["sup_lattice"]):
        out[str(a)] = {"slope_continuous": float(np.polyfit(nd, np.log(c), 1)[0]),
                       "slope_lattice": float(np.polyfit(nd, np.log(l), 1)[0])}
    return out


if __name__ == "__main__":
    res = {"A_base_walk": part_a(), "A_base_walk_weibull": part_a_weibull(),
           "B_third_cumulant": part_b(),
           "C_increment_family": part_c(), "D_ig_rate": part_d(),
           #  Weibull skewness tends to that of the Gumbel minimum as the shape
           #  grows, so a moment-matched Weibull never becomes normal
           "weibull_skewness": {
               "by_shape": {str(k): float(stats.weibull_min(k).stats(moments="s"))
                            for k in (3.6, 10, 100, 1000)},
               "limit": float(-12 * np.sqrt(6) * special.zeta(3) / np.pi ** 3)}}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("B", [(r["n"], round(r["skewness"], 4)) for r in res["B_third_cumulant"]])
    print("D", {k: (round(v["slope_continuous"], 2), round(v["slope_lattice"], 2))
               for k, v in res["D_ig_rate"].items()})
    print("  ->", OUT)
