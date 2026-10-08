# -*- coding: utf-8 -*-
r"""tab_realdata.tex from results/realdata/realdata.json (study_realdata.py)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
r = json.load(open(os.path.join(RES, "realdata", "realdata.json")))

L, A = r["R1_laser"], r["R1_alloy_detrended"]
RA, RW = r["R2_alloy"], r["R2_wear"]
P = r["R3_fatigue_life"]
C = json.load(open(os.path.join(RES, "realdata", "alloy_clock.json")))["sheppard"]
DB = json.load(open(os.path.join(RES, "realdata", "deviceb_clock.json")))["all"]

lines = [
    r"\begin{table}[htbp]", r"\centering\small",
    r"\caption{Real data \citep{MeekerEscobar1998}. Upper: exponent $b$ of the"
    r" within-unit variance of an $m$-step increment, $\mathrm{Var}\propto m^b$"
    r" (on a clock, $\propto S^b$ with $S$ the clock length),"
    r" with $95\,\%$ bootstrap interval; consistency predicts $b=1$, interval"
    r" scaling $b=2$. Middle: unit-by-unit likelihood, $\Delta\ell$ consistent"
    r" minus interval scaling, $p$ a one-sided sign test. Lower: $\Delta$AIC of"
    r" fatigue-life families.}",
    r"\label{tab:realdata}",
    #  fixed widths for the text columns: as l-columns the result strings ran
    #  71pt past \linewidth
    #  ragged right: justified narrow cells spread "Alloy-A,  Paris-coupled"
    r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.33\linewidth} r"
    r" >{\raggedright\arraybackslash}p{0.52\linewidth}@{}}", r"\toprule",
    r"data set & units & result\\", r"\midrule",
    r"\multicolumn{3}{@{}l}{\emph{variance scaling}}\\",
    f"GaAs laser & {L['units']} & $b={L['b']:.2f}$ "
    f"$[{L['b_lo']:.2f},\\,{L['b_hi']:.2f}]$\\\\",
    f"Alloy-A crack length, detrended & {A['units']} & $b={A['b']:.2f}$ "
    f"$[{A['b_lo']:.2f},\\,{A['b_hi']:.2f}]$\\\\",
    f"Alloy-A, damage clock, rounding removed & {A['units']} & $b={C['b']:.2f}$ "
    f"$[{C['b_lo']:.2f},\\,{C['b_hi']:.2f}]$\\\\",
    f"Device-B, clock $t^p$ & {DB['units']} & $b={DB['b']:.2f}$ "
    f"$[{DB['b_lo']:.2f},\\,{DB['b_hi']:.2f}]$\\\\",
    r"\multicolumn{3}{@{}l}{\emph{likelihood, unit by unit}}\\",
    f"Alloy-A, Paris-coupled scale & {RA['units']} & consistent better in "
    f"{RA['consistent_better']}; median $\\Delta\\ell={RA['median_dll']:+.2f}$, "
    f"$p={RA['sign_test_p']:.2f}$\\\\",
    f"metal wear, irregular grid & {RW['units']} & consistent better in "
    f"{RW['consistent_better']}; median $\\Delta\\ell={RW['median_dll']:+.2f}$, "
    f"$p={RW['sign_test_p']:.2f}$\\\\",
    r"\multicolumn{3}{@{}l}{\emph{fatigue lives, $\Delta$AIC}}\\",
    "Yokobori steel & 63 & " + ", ".join(
        f"{k.replace('Birnbaum--Saunders', 'BS').replace('inverse Gaussian', 'IG')} "
        f"${v['delta_aic']:.1f}$" for k, v in
        sorted(P.items(), key=lambda kv: kv[1]['delta_aic'])) + r"\\",
    r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(RES, "tables", "tab_realdata.tex"), "w", encoding="utf-8",
     newline="\n").write("\n".join(lines))
print("  -> results/tables/tab_realdata.tex")
