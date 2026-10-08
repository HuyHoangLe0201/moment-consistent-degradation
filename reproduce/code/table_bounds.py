# -*- coding: utf-8 -*-
r"""tab_bounds.tex from results/theory/bounds.json (study_bounds.py): the
explicit bounds of Theorems B.1 and B.2 beside the errors measured by simulation."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
B = json.load(open(os.path.join(RES, "theory", "bounds.json")))

ALPHAS, NDS = (0.2, 0.4, 0.6), (10, 25, 50, 100)
hom = {(round(r["alpha"], 1), int(r["n_d"])): r for r in B["homogeneous"]}
rows_a = ""
for a in ALPHAS:
    cells = " & ".join(f"{hom[(a, n)]['law_bound']:.2f} ({hom[(a, n)]['law_measured']:.3f})"
                       for n in NDS)
    mean = " & ".join(f"{hom[(a, n)]['mean_bound_pct']:.1f}" for n in (50, 100))
    rows_a += f"{a:.1f} & {cells} & {mean}\\\\\n"
rows_b = ""
for r in B["coupled"]:
    if abs(r["alpha"] - 0.4) > 1e-9:
        continue
    rows_b += (f"{r['gamma']:.1f} & {r['x_ref']:.2f} & {r['n_d']:.0f} & {r['eps']:.1e} & "
               f"{r['mean_bound_pct']:.1f} & {r['mean_measured_pct']:.2f} & "
               f"{r['sd_pred_steps']:.2f} & {r['sd_bound_steps']:.2f} & "
               f"{r['sd_measured_pct'] * r['sd_pred_steps'] / 100:.3f}\\\\\n")

tex = (r"""\begin{table}[htbp]
\centering\footnotesize
\caption{Explicit bounds against measured errors. Upper: \Cref{thm:boundhom}, BS
steps of shape $\alpha$; law bound with the measured distance of
\Cref{tab:igapprox} in brackets, and the mean bound in \% of $n_d$. Lower:
\Cref{thm:boundcpl} for the settings of \Cref{tab:lamperti} with $\alpha=0.4$;
mean bound and measured error in \% of the predicted mean; s.d.\ bound of
\Cref{prop:sdcpl} and measured s.d.\ error, in steps.}
\label{tab:bounds}
\begin{tabular}{@{}c cccc cc@{}}
\toprule
& \multicolumn{4}{c}{$\sup_t|P(\tau\le t)-F_{\IG}(t)|$, bound (measured)}
& \multicolumn{2}{c}{mean bound [\%]}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
$\alpha$ & $n_d=10$ & $25$ & $50$ & $100$ & $n_d=50$ & $100$\\
\midrule
""" + rows_a + r"""\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}cc r c rr rrr@{}}
\toprule
& & & & \multicolumn{2}{c}{mean [\%]} & \multicolumn{3}{c}{s.d.\ [steps]}\\
\cmidrule(lr){5-6}\cmidrule(lr){7-9}
$\gamma$ & $x_{\rm ref}$ & $n_d$ & $\bar\varepsilon$ & bound & measured
 & predicted & bound & measured\\
\midrule
""" + rows_b + r"""\bottomrule
\end{tabular}
\end{table}
""")
open(os.path.join(RES, "tables", "tab_bounds.tex"), "w", encoding="utf-8",
     newline="\n").write(tex)
print("  -> results/tables/tab_bounds.tex")
