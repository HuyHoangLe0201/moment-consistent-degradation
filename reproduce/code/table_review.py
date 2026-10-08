# -*- coding: utf-8 -*-
r"""Two tables from results/theory/review_extras.json (study_review.py).

tab_sumlaw.tex    law-level accuracy: a BS random walk on a base grid, its
                  k-step increment against the moment-matched BS law of
                  Theorem 1 and against interval scaling
tab_incfamily.tex increment families on the real data, all in the
                  moment-consistent form of Proposition 2
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
r = json.load(open(os.path.join(RES, "theory", "review_extras.json"), encoding="utf-8"))

# ── law-level accuracy ──────────────────────────────────────────────────────
KS = (2, 4, 16, 64)
g_bs = lambda a: a * a * (1 + 1.25 * a * a) / (1 + 0.5 * a * a) ** 2
rows = []
for fam, A, key in (("BS", r["A_base_walk"], "alpha_ref"),
                    ("Weibull", r["A_base_walk_weibull"], "alpha_equiv")):
    for a in sorted({x[key] for x in A}):
        cells = {x["k"]: x for x in A if x[key] == a}
        s0 = cells[KS[0]]["s0"]
        rows.append(f"{fam} & {g_bs(a):.3f} & $1/{round(1 / s0)}$ & "
                    + " & ".join(f"{cells[k]['sup_consistent']:.3f}" for k in KS) + " & "
                    + " & ".join(f"{cells[k]['sup_interval']:.2f}" for k in KS) + r"\\")
    if fam == "BS":
        rows.append(r"\addlinespace")
ks = " & ".join(f"${k}$" for k in KS)
tab = [r"\begin{table}[htbp]", r"\centering\small",
       r"\caption{Supremum distance between the exact law of $k$ summed base"
       r" increments of a random walk and two single-law descriptions of it: the"
       r" moment-consistent law of the same family and interval scaling. $c^2$ is"
       r" the squared coefficient of variation per unit time, equal for the two"
       r" families; the BS base step $s_0$ is admissible by \Cref{rem:finegrid}.}",
       r"\label{tab:sumlaw}",
       r"\begin{tabular}{@{}lcc cccc cccc@{}}", r"\toprule",
       r"& & & \multicolumn{4}{c}{moment-consistent law} & \multicolumn{4}{c}{interval scaling}\\",
       r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
       r"family & $c^2$ & $s_0$ & " + ks + " & " + ks + r"\\", r"\midrule",
       *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(RES, "tables", "tab_sumlaw.tex"), "w", encoding="utf-8",
     newline="\n").write("\n".join(tab))

# ── increment families on the data ──────────────────────────────────────────
C = r["C_increment_family"]
FAMS = ("BS", "gamma", "inverse Gaussian", "lognormal", "Weibull")
SHORT = {"BS": "BS", "gamma": "gamma", "inverse Gaussian": "IG",
         "lognormal": "lognormal", "Weibull": "Weibull"}
LABEL = {"laser": "laser", "metal wear": "metal wear",
         "metal wear, no merged steps": r"\quad no merged steps",
         "Alloy-A": "Alloy-A", "Alloy-A, rounding-aware": r"\quad rounding-aware"}
rows = []
for name in ("laser", "metal wear", "metal wear, no merged steps", "Alloy-A",
             "Alloy-A, rounding-aware"):
    d = C[name]
    lo, _, hi = d["boot_dAIC_BS_minus_Weibull"]
    rows.append(f"{LABEL[name]} ({d['units']}) & "
                + " & ".join(f"{d['delta_aic'][f]:.1f} ({d['units_best'][f]})" for f in FAMS)
                + f" & $[{lo:.0f},\\,{hi:.0f}]$" + r"\\")
tab = [r"\begin{table}[htbp]", r"\centering\footnotesize",
       r"\caption{Increment law on the real data. Every family is put in the"
       r" moment-consistent form of \Cref{prop:general} and fitted unit by unit by"
       r" maximum likelihood with the same number of parameters. Entries:"
       r" $\Delta$AIC summed over units (units where the family fits best). Last"
       r" column: $95\,\%$ bootstrap interval over units of $\Delta$AIC, BS minus"
       r" Weibull (W). Indented rows: wear without the two units that needed a merged"
       r" step; Alloy-A with a likelihood that accounts for the $0.01$-inch rounding.}",
       r"\label{tab:incfamily}",
       r"\begin{tabular}{@{}l ccccc c@{}}", r"\toprule",
       r"data set (units) & " + " & ".join(SHORT[f] for f in FAMS) + r" & BS$-$W\\",
       r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(RES, "tables", "tab_incfamily.tex"), "w", encoding="utf-8",
     newline="\n").write("\n".join(tab))
# ── first-passage approximations: BS against Weibull steps ──────────────────
TH = os.path.join(RES, "theory")
W = json.load(open(os.path.join(TH, "weibull_check.json"), encoding="utf-8"))
T2 = json.load(open(os.path.join(TH, "theory_results.json"), encoding="utf-8"))["T2"]
LB = json.load(open(os.path.join(TH, "lamperti.json"), encoding="utf-8"))["grid"]
LW = W["lamperti"]


def col(tab, nds, nd):
    j = nds.index(nd)
    return max(row[j] for row in tab)


def sd_bs(r):          # BS table stores simulated over predicted; flip it
    return 100 * (1 / (1 + r["sd_err_pct_2nd"] / 100) - 1)


lines = [
    (r"IG approximation, $\sup$ distance, $n_d=10$",
     col(T2["sup_continuous"], T2["n_d"], 10), col(W["ig"]["sup_continuous"], W["ig"]["n_d"], 10), "{:.3f}"),
    (r"IG approximation, $\sup$ distance, $n_d=50$",
     col(T2["sup_continuous"], T2["n_d"], 50), col(W["ig"]["sup_continuous"], W["ig"]["n_d"], 50), "{:.3f}"),
    (r"Lamperti, first order, $|$mean error$|$ [\%]",
     max(abs(r["mean_err_pct_1st"]) for r in LB), max(abs(r["mean_err_pct_1st"]) for r in LW), "{:.1f}"),
    (r"Lamperti, second order, $|$mean error$|$ [\%]",
     max(abs(r["mean_err_pct_2nd"]) for r in LB), max(abs(r["mean_err_pct_2nd"]) for r in LW), "{:.2f}"),
    (r"Lamperti, second order, $|$s.d.\ error$|$ [\%]",
     max(abs(sd_bs(r)) for r in LB), max(abs(r["sd_overpred_pct_dV"]) for r in LW), "{:.2f}"),
    (r"Lamperti, second order, $\sup$ distance",
     max(r["sup_2nd"] for r in LB), max(r["sup_2nd"] for r in LW), "{:.3f}"),
]
rows = [f"{lab} & {fmt.format(b)} & {fmt.format(w)}\\\\" for lab, b, w, fmt in lines]
tab = [r"\begin{table}[htbp]", r"\centering\small",
       r"\caption{The first-passage approximations with BS and with Weibull steps of"
       r" the same mean and variance. Each entry is the worst case over the shapes"
       r" of \Cref{tab:igapprox} or the sixteen settings of \Cref{tab:lamperti}.}",
       r"\label{tab:weibcheck}",
       r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
       r"check & BS & Weibull\\", r"\midrule", *rows,
       r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(RES, "tables", "tab_weibcheck.tex"), "w", encoding="utf-8",
     newline="\n").write("\n".join(tab))
print("  -> results/tables/tab_sumlaw.tex, tab_incfamily.tex, tab_weibcheck.tex")
