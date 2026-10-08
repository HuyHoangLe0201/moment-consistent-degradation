"""Every table of the paper, from the JSON results.

Writes results/tables/*.tex.

    python code/make_tables.py
"""
from __future__ import annotations

import json
import os
import re
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
TH = os.path.join(HERE, "..", "results", "theory")
OUT = os.path.join(HERE, "..", "results", "tables")
os.makedirs(OUT, exist_ok=True)


def jload(p):
    with open(p) as f:
        return json.load(f)


def _fit(s):
    """Wrap every tabular so it shrinks to the text width instead of overrunning.

    elsarticle's measure is narrower than the article class these tables were
    first laid out for, and several of them ran up to 70pt into the margin.
    adjustbox only scales when a table is actually too wide, so tables that
    already fit are left at their natural size and unscaled.
    """
    return re.sub(
        r"(\\begin\{tabular\}.*?\\end\{tabular\})",
        r"\\begin{adjustbox}{max width=\\linewidth}\n\1\n\\end{adjustbox}",
        s, flags=re.S)


def write(name, s):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(_fit(s))
    print(f"  -> results/tables/{name}")



def t_grid():
    d = jload(os.path.join(TH, "grid_consistency.json"))
    rows = ""
    for a, b in zip(d["naive"], d["consistent"]):
        rows += (f"{a['n']} & {20.0/a['n']:.3f} & {a['mean']:.3f} & {a['var']:.4f} & "
                 f"{b['mean']:.3f} & {b['var']:.4f} & {b['alpha_s']:.4f} & "
                 f"{b['beta_s']:.4f}\\\\\n")
    fp = d["first_passage"]
    frows = ""
    for a, b in zip(fp["naive"], fp["consistent"]):
        frows += (f"{a['steps_per_unit']} & {a['mean']:.3f} & {a['sd']:.3f} & "
                  f"{b['mean']:.3f} & {b['sd']:.3f}\\\\\n")
    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{Grid dependence of the BS increment process: $\\alpha=0.4$, $\\beta=1$,
$\\Dt_{{\\rm ref}}=1$, horizon $T=20$ split into $n$ steps. Lower panel: simulated
first passage to
$D=12$, $4\\times10^4$ paths per grid.}}
\\label{{tab:grid}}
\\begin{{tabular}}{{@{{}}rr rr rr rr@{{}}}}
\\toprule
& & \\multicolumn{{2}}{{c}}{{naive \\eqref{{eq:naive}}}}
  & \\multicolumn{{2}}{{c}}{{grid-consistent}}
  & \\multicolumn{{2}}{{c}}{{parameters}}\\\\
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}\\cmidrule(lr){{7-8}}
$n$ & $s=T/n$ & $\\mathbb{{E}}[X(T)]$ & $\\mathrm{{Var}}[X(T)]$
    & $\\mathbb{{E}}[X(T)]$ & $\\mathrm{{Var}}[X(T)]$ & $\\alpha_s$ & $\\beta_s$\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}

\\vspace{{6pt}}
\\begin{{tabular}}{{@{{}}r rr rr@{{}}}}
\\toprule
& \\multicolumn{{2}}{{c}}{{naive}} & \\multicolumn{{2}}{{c}}{{grid-consistent}}\\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}
steps per unit time & $\\mathbb{{E}}[\\tau]$ & $\\mathrm{{sd}}(\\tau)$
                    & $\\mathbb{{E}}[\\tau]$ & $\\mathrm{{sd}}(\\tau)$\\\\
\\midrule
{frows}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# ── IG approximation accuracy ──────────────────────────────────────────────
def t_ig():
    d = jload(os.path.join(TH, "theory_results.json"))["T2"]
    rows = ""
    for i, a in enumerate(d["alphas"]):
        c = " & ".join(f"{v:.4f}" for v in d["sup_continuous"][i])
        l = " & ".join(f"{v:.3f}" for v in d["sup_lattice"][i])
        rows += f"{a:.1f} & {c} & {l}\\\\\n"
    nd = " & ".join(str(n) for n in d["n_d"])
    k = len(d["n_d"])
    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{Accuracy $\\sup_\\tau|F_{{\\RUL}}-F_{{\\IG}}|$ of \\Cref{{thm:rulmulti}},
$6\\times10^4$ simulated first passages per cell, by shape $\\alpha$ and expected
remaining increments $n_d$ (columns). Left: interpolated crossing instant; right:
crossing index (\\Cref{{rem:lattice}}).}}
\\label{{tab:igapprox}}
\\begin{{tabular}}{{@{{}}c{'c'*k}{'c'*k}@{{}}}}
\\toprule
& \\multicolumn{{{k}}}{{c}}{{continuous crossing instant}}
& \\multicolumn{{{k}}}{{c}}{{crossing index (lattice)}}\\\\
\\cmidrule(lr){{2-{k+1}}}\\cmidrule(lr){{{k+2}-{2*k+1}}}
$\\alpha$ & {nd} & {nd}\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# ── Lamperti reduction ─────────────────────────────────────────────────────
def t_lamperti():
    d = jload(os.path.join(TH, "lamperti.json"))
    rows = ""
    for r in d["grid"]:
        rows += (f"{r['gamma']:.1f} & {r['x_ref']:.2f} & {r['alpha']:.1f} & "
                 f"{r['n_d']:.0f} & {r['sup_1st']:.4f} & {r['sup_2nd']:.4f} & "
                 f"{r['mean_err_pct_1st']:+.2f} & {r['mean_err_pct_2nd']:+.2f} & "
                 f"{r['sd_overpred_pct_dZ']:+.1f} & "
                 #  sd_err_pct_2nd is simulated over predicted; turn it into
                 #  predicted over simulated, the convention of the d_Z column
                 #  beside it, or equal predictions at gamma = 0 show opposite signs
                 f"{100 * (1 / (1 + r['sd_err_pct_2nd'] / 100) - 1):+.2f}\\\\\n")
    crows = ""
    for r in d["coarseness"]:
        crows += (f"{r['beta_over_xref']:.3f} & {r['n_d']:.0f} & {r['sup']:.4f} & "
                  f"{r['mean_err_pct']:+.2f}\\\\\n")
    worst1 = max(r["sup_1st"] for r in d["grid"])
    worst2 = max(r["sup_2nd"] for r in d["grid"])
    wmean1 = max(abs(r["mean_err_pct_1st"]) for r in d["grid"])
    wmean2 = max(abs(r["mean_err_pct_2nd"]) for r in d["grid"])
    wsd2 = max(abs(r["sd_err_pct_2nd"]) for r in d["grid"])
    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{Accuracy of \\Cref{{thm:lamperti}}, $4\\times10^4$ first passages per row,
$\\beta=0.02$, $D=3$. First order drops $\\kappa$ and $d_V$. Mean error: simulated
over predicted, minus one; s.d.\\ error: predicted over simulated, minus one, with
$\\kappa$ kept. Lower panel: step coarseness at $\\gamma=1.5$, $\\alpha=0.3$.}}
\\label{{tab:lamperti}}
\\begin{{tabular}}{{@{{}}ccc r rr rr rr@{{}}}}
\\toprule
& & & & \\multicolumn{{2}}{{c}}{{$\\sup_\\tau|F-F_{{\\IG}}|$}}
      & \\multicolumn{{2}}{{c}}{{mean RUL error [\\%]}}
      & \\multicolumn{{2}}{{c}}{{s.d.\\ error [\\%]}}\\\\
\\cmidrule(lr){{5-6}}\\cmidrule(lr){{7-8}}\\cmidrule(lr){{9-10}}
$\\gamma$ & $x_{{\\rm ref}}$ & $\\alpha$ & $n_d$
 & first order & second order & first order & second order & over $d_Z$ & over $d_V$\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}

\\vspace{{6pt}}
\\begin{{tabular}}{{@{{}}rrrr@{{}}}}
\\toprule
$\\beta/x_{{\\rm ref}}$ & $n_d$ & $\\sup_\\tau|F-F_{{\\IG}}|$ & mean RUL error [\\%]\\\\
\\midrule
{crows}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


if __name__ == "__main__":
    write("tab_grid.tex", t_grid())
    write("tab_igapprox.tex", t_ig())
    write("tab_lamperti.tex", t_lamperti())
    for name in ("table_singlerate", "table_realdata", "table_review",
                 "table_application", "table_bounds"):
        try:
            runpy.run_path(os.path.join(HERE, name + ".py"), run_name="__main__")
        except FileNotFoundError as e:
            print(f"  skip {name}: {os.path.basename(str(e.filename))} missing "
                  "(run fetch_data.py and the data steps first)")
