"""Calibration curves of the end-to-end application (Section 7.2).

For every nominal level p, the empirical coverage of the central p-interval of
the predictive law is the share of forecasts whose PIT lies in
[(1-p)/2, (1+p)/2].  Rounded readings (Alloy-A) give a PIT interval
[pit_lo, pit_hi]; its midpoint is used.  Bands: 95 % pointwise intervals from
resampling units (forecasts of one unit share its path).  BS steps.

Reads results/realdata/application*.json (no new simulation) and writes the
TikZ figure results/figures/fig_calibration.tex and the summary
results/realdata/calibration.json.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RD = os.path.join(ROOT, "results", "realdata")
OUT_F = os.path.join(ROOT, "results", "figures", "fig_calibration.tex")
OUT_J = os.path.join(RD, "calibration.json")

SETS = [("application.json", "GaAs lasers"), ("application_deviceb.json", "Device-B"),
        ("application_alloy.json", "Alloy-A")]
P = np.round(np.arange(0.05, 0.951, 0.05), 2)
NB, SEED = 2000, 7
W, H, GAP = 3.4, 3.4, 1.0                      # panel size and gap, cm


def pit(r):
    if "pit" in r:
        return r["pit"]
    return 0.5 * (r["pit_lo"] + r["pit_hi"])


def curve(levels):
    u = np.array([r["unit"] for r in levels])
    z = np.abs(np.array([pit(r) for r in levels]) - 0.5)
    cov = np.array([np.mean(z <= p / 2 + 1e-12) for p in P])
    units = np.unique(u)
    rng = np.random.default_rng(SEED)
    idx = {k: np.nonzero(u == k)[0] for k in units}
    boot = np.empty((NB, len(P)))
    for b in range(NB):
        pick = np.concatenate([idx[k] for k in rng.choice(units, len(units))])
        boot[b] = [np.mean(z[pick] <= p / 2 + 1e-12) for p in P]
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return cov, lo, hi


def main():
    summary = []
    panels = []
    for f, label in SETS:
        data = [r for r in json.load(open(os.path.join(RD, f))) if r["law"] == "BS"]
        fine = next(r for r in data if r["h_dep"] < r["h_ref"])
        coarse = next(r for r in data if r["h_dep"] > r["h_ref"])
        pan = {"label": label}
        for tag, r in (("c2f", fine), ("f2c", coarse)):
            for m in ("consistent", "interval"):
                L = r[m]["levels"]
                if "pit" not in L[0] and "pit_lo" not in L[0]:
                    raise SystemExit(f"{f}: no PIT stored; rerun the study")
                cov, lo, hi = curve(L)
                pan[(tag, m)] = (cov, lo, hi)
                summary.append({"data": label, "direction": tag, "model": m,
                                "n": len(L), "mace": float(np.mean(np.abs(cov - P))),
                                "coverage": cov.tolist(), "lo": lo.tolist(),
                                "hi": hi.tolist(), "p": P.tolist()})
        panels.append(pan)
    json.dump(summary, open(OUT_J, "w"), indent=1)
    for s in summary:
        print(f"{s['data']:12s} {s['direction']} {s['model']:10s} n {s['n']:3d}"
              f"  mean |cov - p| {s['mace']:.3f}")
    write_tikz(panels)


def pts(x, y, x0):
    return " ".join(f"({x0 + W * a:.3f},{H * b:.3f})" for a, b in zip(x, y))


def write_tikz(panels):
    L = [r"\begin{figure}[tbp]", r"\centering",
         r"\begin{tikzpicture}[font=\scriptsize,line cap=round,line join=round]",
         r"\definecolor{cons}{rgb}{0.122,0.306,0.475}\definecolor{intv}{rgb}{0.753,0.314,0.302}"
         r"\definecolor{ink}{rgb}{0.20,0.20,0.22}"]
    for j, pan in enumerate(panels):
        x0 = j * (W + GAP)
        L.append(rf"\node[anchor=south west,font=\footnotesize,inner sep=0] at ({x0:.2f},{H + 0.15:.2f}) {{({chr(97 + j)}) {pan['label']}}};")
        for m, col in (("consistent", "cons"), ("interval", "intv")):
            cov, lo, hi = pan[("c2f", m)]
            band = (pts(P, hi, x0) + " " + pts(P[::-1], lo[::-1], x0)).replace(") (", ") -- (")
            L.append(rf"\fill[{col}!15] {band} -- cycle;")
        L.append(rf"\draw[ink!60,densely dotted] ({x0:.3f},0) -- ({x0 + W:.3f},{H:.3f});")
        for m, col in (("consistent", "cons"), ("interval", "intv")):
            L.append(rf"\draw[{col},line width=0.9pt] plot coordinates {{{pts(P, pan[('c2f', m)][0], x0)}}};")
            L.append(rf"\draw[{col},line width=0.7pt,dashed] plot coordinates {{{pts(P, pan[('f2c', m)][0], x0)}}};")
        L.append(rf"\draw[ink,line width=0.45pt] ({x0:.3f},0) rectangle ({x0 + W:.3f},{H:.3f});")
        for t in (0, 0.5, 1):
            L.append(rf"\draw[ink,line width=0.45pt] ({x0 + W * t:.3f},0) -- ++(0,-0.07) node[below,inner sep=1.5pt] {{{t:g}}};")
            if j == 0:
                L.append(rf"\draw[ink,line width=0.45pt] ({x0:.3f},{H * t:.3f}) -- ++(-0.07,0) node[left,inner sep=1.5pt] {{{t:g}}};")
        L.append(rf"\node[anchor=north] at ({x0 + W / 2:.3f},-0.38) {{nominal level}};")
    L.append(rf"\node[rotate=90,anchor=south] at (-0.55,{H / 2:.3f}) {{empirical coverage}};")
    xm = (3 * W + 2 * GAP) / 2                 # legend centred under the panels
    L += [rf"\draw[cons,line width=0.9pt] ({xm - 4.4:.2f},-1.0) -- ++(0.45,0) node[right,text=ink] {{consistent}};",
          rf"\draw[intv,line width=0.9pt] ({xm - 1.6:.2f},-1.0) -- ++(0.45,0) node[right,text=ink] {{interval scaling}};",
          rf"\draw[ink,line width=0.7pt,dashed] ({xm + 1.7:.2f},-1.0) -- ++(0.45,0) node[right,text=ink] {{reverse change}};"]
    L += [r"\end{tikzpicture}",
          r"\caption{Calibration of the end-to-end application, BS steps: empirical"
          r" coverage of central predictive intervals against their nominal level."
          r" Solid, coarse records and fine monitoring, with $95\,\%$ unit-bootstrap"
          r" bands; dashed, the reverse.}",
          r"\label{fig:calibration}", r"\end{figure}", ""]
    os.makedirs(os.path.dirname(OUT_F), exist_ok=True)
    open(OUT_F, "w", encoding="utf-8", newline="\n").write("\n".join(L))


if __name__ == "__main__":
    main()
