"""Sensitivity of the replacement decision to the coupling parameters (gamma, x_ref).

The asset follows the consistent coupled BS process (eq. pcbs) with gamma0 = 1,
x_ref0 = 1, beta = 0.02, threshold D = 3, dt = dt_ref = 1 -- the setting of the
Lamperti table -- at two step-noise levels, alpha = 0.3 and 1.0.  An analyst
uses a wrong (gamma, x_ref).  Fitted to run-to-failure data, the analyst's model
would reproduce the mean life of a new unit, so the step scale is recalibrated
(alpha kept, so the squared CV of a reference step is kept) until the predicted
mean life from x = 0 equals the true one.  At the mid-life state x_k = 1.5 the
analyst replaces at the 5 % quantile of the predicted RUL law; we report the
true probability of failing first, and the predicted mean and s.d. of the RUL
relative to the truth.

Laws are the closed-form IG approximation of the paper (eq. rulcoupled); the
true risk is simulated from the true process.  Writes
results/theory/sensitivity.json and results/tables/tab_sensitivity.tex.
"""
import json
import os

import numpy as np
from scipy import optimize, stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_J = os.path.join(ROOT, "results", "theory", "sensitivity.json")
OUT_T = os.path.join(ROOT, "results", "tables", "tab_sensitivity.tex")

BETA, D, ETA = 0.02, 3.0, 0.05
TRUE = (1.0, 1.0)                                    # (gamma, x_ref)
CASES = [(0.5, 1.0), (0.75, 1.0), (0.9, 1.0), (1.0, 1.0), (1.1, 1.0), (1.25, 1.0),
         (1.5, 1.0), (1.0, 0.8), (1.0, 1.25)]
ALPHAS = (0.3, 1.0)                                  # low and high step noise
X_K = 1.5                                            # mid-life state
N_SIM, SEED = 20000, 11


def bs_moments(a, b):
    return b * (1 + a * a / 2), b * b * a * a * (1 + 1.25 * a * a)


def psi(x, g, xr):
    q = 1 + x / xr
    if g == 0:
        return x
    if abs(g - 1) < 1e-12:
        return xr * np.log(q)
    return xr / (1 - g) * (q ** (1 - g) - 1)


def ig_params(x, mu, var, g, xr):
    """(mean, lambda) of the IG law of the RUL from x (eq. rulcoupled)."""
    qx, qD = 1 + x / xr, 1 + D / xr
    kap = 0.0 if g == 0 else (var / (2 * mu) * (qx ** -g - qD ** -g)
                              + g * mu / 2 * np.log(qD / qx))
    dmu = psi(D, g, xr) - psi(x, g, xr) + kap
    dv = psi(D, 2 * g, xr) - psi(x, 2 * g, xr)
    return dmu / mu, dmu ** 3 / (var * dv)


def ig_cdf(t, m, lam):
    return stats.invgauss.cdf(t, m / lam, scale=lam)


def ig_ppf(p, m, lam):
    return stats.invgauss.ppf(p, m / lam, scale=lam)


def calibrated_moments(alpha, g, xr, life):
    """Step moments with alpha fixed whose predicted mean life from 0 is `life`."""
    def f(lc):
        mu, var = bs_moments(alpha, BETA * np.exp(lc))
        return ig_params(0.0, mu, var, g, xr)[0] - life
    lc = optimize.brentq(f, -5, 5)
    return bs_moments(alpha, BETA * np.exp(lc))


def bs_step(mu, var, rng, size):
    """Consistent BS step with the given mean and variance (closed form)."""
    r = var / mu ** 2
    u = np.sqrt(1 + 3 * r)
    A = 2 * r * (u + 4) / ((5 - r) * (u + 1))
    b = mu / (1 + A / 2)
    z = rng.standard_normal(size)
    h = np.sqrt(A) * z / 2
    return b * (h + np.sqrt(h * h + 1)) ** 2


def simulate_tau(x0, mu, var, g, xr, n, rng):
    """Interpolated first passage of the true coupled process from x0."""
    x = np.full(n, x0)
    tau = np.full(n, np.nan)
    alive = np.ones(n, bool)
    k = 0
    while alive.any():
        idx = np.nonzero(alive)[0]
        s = (1 + x[idx] / xr) ** g
        dx = bs_step(s * mu, s * var, rng, idx.size)
        xn = x[idx] + dx
        hit = xn >= D
        tau[idx[hit]] = k + (D - x[idx][hit]) / dx[hit]
        x[idx] = xn
        alive[idx[hit]] = False
        k += 1
    return tau


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for alpha in ALPHAS:
        mu0, var0 = bs_moments(alpha, BETA)
        life = ig_params(0.0, mu0, var0, *TRUE)[0]
        tau = simulate_tau(X_K, mu0, var0, *TRUE, N_SIM, rng)
        mt, lt = ig_params(X_K, mu0, var0, *TRUE)
        for g, xr in CASES:
            mu, var = calibrated_moments(alpha, g, xr, life)
            ma, la = ig_params(X_K, mu, var, g, xr)
            q = ig_ppf(ETA, ma, la)
            rows.append({
                "alpha": alpha, "gamma": g, "x_ref": xr, "x": X_K,
                "true_cv": float(np.sqrt(mt / lt)),
                "mean_ratio": float(ma / mt),
                "sd_ratio": float(np.sqrt(ma ** 3 / la) / np.sqrt(mt ** 3 / lt)),
                "risk_ig": float(ig_cdf(q, mt, lt)),
                "risk_sim": float(np.mean(tau <= q)), "q": float(q)})
            r = rows[-1]
            print(f"a {alpha:3.1f} g {g:4.2f} xr {xr:4.2f}  cv {r['true_cv']:.3f}"
                  f"  mean {r['mean_ratio']:.3f}  sd {r['sd_ratio']:.3f}"
                  f"  risk IG {100 * r['risk_ig']:5.1f}%  sim {100 * r['risk_sim']:5.1f}%")
    os.makedirs(os.path.dirname(OUT_J), exist_ok=True)
    json.dump({"beta": BETA, "D": D, "eta": ETA, "true": TRUE, "x_k": X_K,
               "n_sim": N_SIM, "rows": rows}, open(OUT_J, "w"), indent=1)
    write_table(rows)


def write_table(rows):
    by = {}
    for r in rows:
        by.setdefault((r["gamma"], r["x_ref"]), {})[r["alpha"]] = r
    body = []
    for (g, xr), d in by.items():
        cells = [f"{d[a]['mean_ratio']:.2f} & {d[a]['sd_ratio']:.2f} & {100 * d[a]['risk_sim']:.1f}"
                 for a in ALPHAS]
        mark = " (true)" if (g, xr) == TRUE else ""
        body.append(f"{g:g} & {xr:g}{mark} & " + " & ".join(cells) + r"\\")
    cv = {a: next(r["true_cv"] for r in rows if r["alpha"] == a) for a in ALPHAS}
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{$\alpha={a:g}$, RUL c.v.\ ${cv[a]:.2f}$}}"
                      for a in ALPHAS)
    cmid = "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(ALPHAS)))
    sub = " & ".join([r"mean & s.d. & risk [\%]"] * len(ALPHAS))
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Sensitivity to the coupling parameters at the mid-life state"
           r" $x_k=1.5$. The asset follows the consistent coupled process with $\gamma=1$,"
           r" $x_{\rm ref}=1$, $\beta=0.02$, $D=3$. The analyst uses the"
           r" $(\gamma,x_{\rm ref})$ of each row, with the step scale recalibrated to the"
           r" true mean life of a new unit. Mean, s.d.: predicted over true RUL mean and"
           r" standard deviation; risk: simulated probability of failing before the"
           r" analyst's $5\,\%$ replacement time; c.v.: true coefficient of variation"
           r" of the RUL.}",
           r"\label{tab:sensitivity}", r"\small",
           r"\begin{tabular}{@{}ll " + "rrr " * len(ALPHAS) + r"@{}}", r"\toprule",
           r"& & " + head + r"\\", cmid,
           r"$\gamma$ & $x_{\rm ref}$ & " + sub + r"\\", r"\midrule"] + body + [
           r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(OUT_T, "w", encoding="utf-8", newline="\n").write("\n".join(tex))


if __name__ == "__main__":
    main()
