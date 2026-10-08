"""Brier score of the replacement rule's event forecasts (Section 7.2).

At every inspection t_j before the observed crossing of a threshold D, the
replacement rule forecasts p_j = P(the reading at t_{j+1} reaches D).  The
event y_j is whether it does.  The Brier score is the mean of (p_j - y_j)^2
over all forecasts, scored here at every inspection up to the crossing (not
only up to the replacement, so both models are scored on the same events).

The fits, the predictive simulation and the protocol are those of the three
application studies, imported unchanged; only the random generator is
replaced by a separate one, so the stored application results are untouched.
BS steps.  Writes results/realdata/brier.json and results/tables/tab_brier.tex.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_application as SA                                   # noqa: E402
import study_application_deviceb as SD                           # noqa: E402
import study_application_alloy as SAA                            # noqa: E402
import study_realdata as SR                                      # noqa: E402

OUT_J = os.path.join(ROOT, "results", "realdata", "brier.json")
OUT_T = os.path.join(ROOT, "results", "tables", "tab_brier.tex")
LAW, MODELS = "BS", ("consistent", "interval")
SEED, NB = 20261009, 2000

#  a separate generator: the application results stay as stored
SA.RNG = np.random.default_rng(SEED)
SD.RNG = np.random.default_rng(SEED + 1)
SAA.RNG = np.random.default_rng(SEED + 2)


def lasers(h_ref, h_dep):
    P = SR.paths(SR.load("gaaslaser"), "unit", "hours", "increase")
    out = []
    for i, (t, x) in enumerate(P):
        cv2_ref, df = SA.cv2_pooled([p for j, p in enumerate(P) if j != i], h_ref,
                                    with_df=True)
        cv2 = {"consistent": cv2_ref * h_ref / h_dep, "interval": cv2_ref}
        tt, xx = SA.on_grid(t, x, h_dep)
        for D in SA.THRESHOLDS:
            T_obs = SA.first_cross(tt, xx, D)
            if not np.isfinite(T_obs):
                continue
            for j in range(1, len(tt) - 1):
                if tt[j] >= T_obs:
                    break
                if tt[j] < 500:
                    continue
                y = bool(tt[j + 1] >= T_obs - 1e-9)
                for m in MODELS:
                    S = SA.simulate(LAW, xx[j], tt[j], h_dep, cv2[m], 1, df)
                    out.append({"unit": i, "D": D, "model": m, "y": y,
                                "p": float(np.mean(S[:, 0] >= D))})
    return out


def deviceb(h_ref, h_dep):
    P = SD.load()
    out = []
    for i, (temp, t, x) in enumerate(P):
        train = [q for j, q in enumerate(P) if j != i and q[0] == temp]
        p = SD.clock_p(train)
        F = SD.fit(train, h_ref, p)
        pu = SD.unit_p(train)
        tt, xx = SD.thin(t, x, h_dep)
        for D in SD.THRESHOLDS:
            hit = np.nonzero(xx >= D)[0]
            if not len(hit):
                continue
            T_obs = float(tt[hit[0]])
            for j in range(1, len(tt) - 1):
                if tt[j] >= T_obs:
                    break
                y = bool(tt[j + 1] >= T_obs - 1e-9)
                for m in MODELS:
                    S = SD.simulate(LAW, m, tt[:j + 1], xx[:j + 1], tt[j + 1:j + 2], p,
                                    F[m], F["df"], pu)
                    out.append({"unit": i, "D": D, "model": m, "y": y,
                                "p": float(np.mean(S[:, 0] >= D))})
    return out


def alloy(h_ref, h_dep):
    P = SR.paths(SR.load("alloya"), "specimen", "megacycles", "inches")
    out = []
    for i, (t, x) in enumerate(P):
        F = SAA.fit([p for j, p in enumerate(P) if j != i], h_ref)
        tt, xx = SAA.thin(t, x, h_dep)
        for D in SAA.THRESHOLDS:
            T_obs = SAA.first_cross(tt, xx, D)
            if not np.isfinite(T_obs):
                continue
            for j in range(1, len(tt) - 1):
                if tt[j] >= T_obs - 1e-9:
                    break
                if tt[j] < 0.02 - 1e-9:
                    continue
                y = bool(tt[j + 1] >= T_obs - 1e-9)
                for m in MODELS:
                    S = SAA.simulate(LAW, m, xx[j], tt[:j + 1], xx[:j + 1], h_dep, h_ref,
                                     F["gamma"], F[m], 1, F["df"], F[m + "_sig2"],
                                     F["gamma_boot"])
                    out.append({"unit": i, "D": D, "model": m, "y": y,
                                "p": float(np.mean(S[:, 0] >= D - 1e-9))})
    return out


SETS = [("GaAs lasers", lasers, SA.DIRECTIONS, "h"),
        ("Device-B", deviceb, SD.DIRECTIONS, "h"),
        ("Alloy-A", alloy, SAA.DIRECTIONS, "kcycles")]


def summarise(rec):
    """Brier score per model, and the consistent-minus-interval difference with
    a 95 % interval from resampling units."""
    rng = np.random.default_rng(SEED)
    by = {m: [r for r in rec if r["model"] == m] for m in MODELS}
    units = np.unique([r["unit"] for r in rec])
    se = {m: {} for m in MODELS}
    for m in MODELS:
        for r in by[m]:
            se[m].setdefault(r["unit"], []).append((r["p"] - r["y"]) ** 2)
    bs = {m: float(np.mean([(r["p"] - r["y"]) ** 2 for r in by[m]])) for m in MODELS}
    diffs = []
    for _ in range(NB):
        pick = rng.choice(units, len(units))
        a = np.concatenate([se["consistent"].get(k, []) for k in pick])
        b = np.concatenate([se["interval"].get(k, []) for k in pick])
        diffs.append(a.mean() - b.mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"brier": bs, "diff": bs["consistent"] - bs["interval"],
            "diff_lo": float(lo), "diff_hi": float(hi),
            "n": len(by["consistent"]), "events": int(sum(r["y"] for r in by["consistent"]))}


def main():
    if "--table" in sys.argv:                  # rewrite the table from the stored results
        write_table(json.load(open(OUT_J)))
        return
    res = []
    for label, fn, dirs, unit in SETS:
        k = 1000.0 if unit == "kcycles" else 1.0
        for h_ref, h_dep in dirs:
            rec = fn(h_ref, h_dep)
            s = summarise(rec)
            s.update({"data": label, "h_ref": h_ref * k, "h_dep": h_dep * k, "unit": unit})
            res.append(s)
            print(f"{label:12s} {h_ref * k:g}->{h_dep * k:g} {unit}: n {s['n']}  events {s['events']}"
                  f"  Brier cons {100 * s['brier']['consistent']:.2f}  int {100 * s['brier']['interval']:.2f}"
                  f"  diff {100 * s['diff']:+.2f} [{100 * s['diff_lo']:+.2f}, {100 * s['diff_hi']:+.2f}]")
    json.dump(res, open(OUT_J, "w"), indent=1)
    write_table(res)


def write_table(res):
    body = []
    for s in res:
        body.append(f"{s['data']} & ${s['h_ref']:g}\\to{s['h_dep']:g}$\\,{s['unit']} & {s['n']} & {s['events']}"
                    f" & {100 * s['brier']['consistent']:.2f} & {100 * s['brier']['interval']:.2f}"
                    f" & ${100 * s['diff']:+.2f}$ [${100 * s['diff_lo']:+.2f}$,\\,${100 * s['diff_hi']:+.2f}$]\\\\")
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Brier score ($\times100$) of the forecasts behind the replacement rule"
           r" of \Cref{sec:application}, BS steps: at each inspection before a crossing,"
           r" the predicted probability that the next reading reaches the threshold."
           r" Schedule: records $\to$ monitoring; $n$: forecasts; events: crossings;"
           r" cons., int.: consistent and interval-scaled models. Last column: their"
           r" difference, with a $95\,\%$ interval that resamples units.}",
           r"\label{tab:brier}", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}ll rr rr l@{}}", r"\toprule",
           r"data & schedule & $n$ & events & cons. & int. & cons.\ $-$ int.\\",
           r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(OUT_T, "w", encoding="utf-8", newline="\n").write("\n".join(tex))


if __name__ == "__main__":
    main()
