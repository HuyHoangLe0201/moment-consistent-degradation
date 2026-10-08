"""Threshold sensitivity of the replacement rule in the end-to-end application.

Reads the stored application results (no new simulation): every replacement
event is kept with its threshold D, so the rule can be scored threshold by
threshold.  Level-interval coverage does not involve the threshold and is not
repeated here.  The two directions of the schedule change share a row.
Writes results/tables/tab_threshold.tex.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RD = os.path.join(ROOT, "results", "realdata")
OUT = os.path.join(ROOT, "results", "tables", "tab_threshold.tex")

SETS = [  # file, label, threshold unit, time unit
    ("application.json", "GaAs lasers", r"\%", "h"),
    ("application_deviceb.json", "Device-B", "dB", "h"),
    ("application_alloy.json", "Alloy-A", "in.", "kcycles"),
]


def score(r, D, k):
    """(crossings, missed cons., missed int., warning cons., warning int.)"""
    out = []
    for m in ("consistent", "interval"):
        Q = [e for e in r[m]["policy"] if e["D"] == D]
        lead = [k * (e["T_obs"] - e["t_rep"]) for e in Q if not e["missed"]]
        out.append((len(Q), sum(e["missed"] for e in Q), np.mean(lead) if lead else np.nan))
    return out[0][0], out[0][1], out[1][1], out[0][2], out[1][2]


def rows_for(fname, label, unit, tu):
    data = [r for r in json.load(open(os.path.join(RD, fname))) if r["law"] == "BS"]
    k = 1000.0 if tu == "kcycles" else 1.0          # alloy times are stored in Mcycles
    fine = next(r for r in data if r["h_dep"] < r["h_ref"])     # coarse records, fine monitoring
    coarse = next(r for r in data if r["h_dep"] > r["h_ref"])
    nd = 1 if k > 1 else 0
    head = (f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{label}: {fine['h_ref'] * k:g}"
            f" $\\leftrightarrow$ {fine['h_dep'] * k:g}\\,{tu}}}}}\\\\")
    out = [head]
    for D in sorted({e["D"] for e in fine["consistent"]["policy"]}):
        cells = []
        for r in (fine, coarse):
            n, mc, mi, lc, li = score(r, D, k)
            cells.append(f"{n} & {mc} & {mi} & {lc:.{nd}f} & {li:.{nd}f}")
        d = f"{D:.2f}" if unit == "in." else f"{D:g}"
        out.append(f"\\quad {d} {unit} & " + " & ".join(cells) + "\\\\")
    return out


def main():
    body = []
    for f, lab, unit, tu in SETS:
        if os.path.exists(os.path.join(RD, f)):
            body += rows_for(f, lab, unit, tu)
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Replacement rule of \Cref{sec:application} threshold by threshold, BS"
           r" steps: observed crossings ($n$), failures missed and mean warning, for the"
           r" consistent (c) and interval-scaled (i) models.}",
           r"\label{tab:threshold}", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}l rrrrr rrrrr@{}}", r"\toprule",
           r"& \multicolumn{5}{c}{coarse records, fine monitoring}"
           r" & \multicolumn{5}{c}{fine records, coarse monitoring}\\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
           r"& & \multicolumn{2}{c}{missed} & \multicolumn{2}{c}{warning}"
           r" & & \multicolumn{2}{c}{missed} & \multicolumn{2}{c}{warning}\\",
           r"threshold & $n$ & c & i & c & i & $n$ & c & i & c & i\\", r"\midrule"]
    tex += body + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(tex))
    print("\n".join(body))


if __name__ == "__main__":
    main()
