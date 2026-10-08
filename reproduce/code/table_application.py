# -*- coding: utf-8 -*-
r"""tab_application.tex from the three application studies:
results/realdata/application.json (lasers), application_deviceb.json and
application_alloy.json.  BS steps only; Weibull steps change no coverage by
more than WMAX points, which the text reports and this script checks."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rd = lambda f: json.load(open(os.path.join(RES, "realdata", f)))
SETS = (("GaAs lasers: hours; level at $4000$\\,h, width in \\%, warning in h",
         rd("application.json"), lambda h: f"{h:.0f}", ".2f", 1.0, ".0f"),
        ("Device-B: hours; level at end of test, width in dB, warning in h",
         rd("application_deviceb.json"), lambda h: f"{h:.0f}", ".2f", 1.0, ".0f"),
        ("Alloy-A: kcycles; length $40$\\,kcycles ahead, width in inches,"
         " warning in kcycles",
         rd("application_alloy.json"), lambda h: f"{1000 * h:.0f}", ".3f", 1000.0, ".1f"))

pc = lambda v: f"{100 * v:.0f}"
WMAX = 0
for _, R, *_ in SETS:
    for rb in R:
        if rb["law"] != "BS":
            continue
        for rw in (r for r in R if r["law"] != "BS" and r["h_ref"] == rb["h_ref"]):
            for m in ("consistent", "interval"):
                for key in ("cover80", "cover90"):
                    WMAX = max(WMAX, abs(round(100 * rb[m][key]) - round(100 * rw[m][key])))
print(f"  largest coverage difference between BS and the other laws: {WMAX} points")

lines = [
    r"\begin{table}[htbp]", r"\centering\footnotesize",
    r"\caption{Application, leave one unit out, BS steps: records on one"
    r" schedule, monitoring on the other. Coverage (\%) of the level intervals,"
    r" with a $95\,\%$ interval for the $90\,\%$ coverage that resamples units"
    r" (forecasts of one unit are dependent), mean $90\,\%$ width, and the"
    r" replacement rule: failures missed and mean warning.}",
    r"\label{tab:application}",
    r"\begin{tabular}{@{}ll rrr rr@{}}", r"\toprule",
    r"records $\to$ monitoring & model & $80\,\%$ & $90\,\%$ [$95\,\%$ CI] & width"
    r" & missed & warning\\"]
for title, R, fmt_h, wfmt, ls, lfmt in SETS:
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{" + title + r"}}\\"]
    first = True
    for r in R:
        if r["law"] != "BS":
            continue
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for m in ("consistent", "interval"):
            s = r[m]
            head = f"${fmt_h(r['h_ref'])}\\to{fmt_h(r['h_dep'])}$" if m == "consistent" else ""
            ci = f"[{pc(s['ci90'][0])},\\,{pc(s['ci90'][1])}]"
            lines.append(f"{head} & {m} & {pc(s['cover80'])} & {pc(s['cover90'])} {ci}"
                         f" & {s['width90']:{wfmt}} & {s['missed']}/{s['n_events']}"
                         f" & {s['mean_lead'] * ls:{lfmt}}\\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(RES, "tables", "tab_application.tex"), "w", encoding="utf-8",
     newline="\n").write("\n".join(lines))
print("  -> results/tables/tab_application.tex")
