r"""Write every figure of the paper as a TikZ file, and a preview document.

The figure generators (make_tikz_mech.py, tikz_data_figs.py) compute every
curve from the model or from results/; this script calls them and writes

    figures/out/fig_<name>.tex     one tikzpicture per figure
    figures/out/captions.txt       the captions, for reference
    figures/preview.tex            a standalone document that shows them all

    python figures/export_figures.py
    cd figures && pdflatex preview.tex
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_tikz_mech as M          # noqa: E402  (puts ../code on sys.path)
import tikz_data_figs as TD         # noqa: E402

FIGS = (("mechvar", M.fig_mechvar), ("mechroot", M.fig_mechroot),
        ("mechrisk", M.fig_mechrisk), ("lamperti", TD.fig_lamperti),
        ("distances", TD.fig_distances), ("grid", TD.fig_grid),
        ("realdata", TD.fig_realdata), ("application", TD.fig_application))

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,tikz,xcolor}
\usetikzlibrary{arrows.meta,calc,patterns,decorations.pathreplacing}
\newcommand{\BS}{\mathrm{BS}}
\newcommand{\IG}{\mathrm{IG}}
\newcommand{\RUL}{\mathrm{RUL}}
\newcommand{\Dt}{\Delta t}
\newcommand{\DX}{\Delta X}
\newcommand{\DZ}{\Delta Z}
\newcommand{\dd}{\mathrm{d}}
\renewcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\mathrm{Var}}
% cross-references of the paper, shown as their labels here
\newcommand{\Cref}[1]{[#1]}
\newcommand{\citep}[1]{[#1]}
\renewcommand{\eqref}[1]{(#1)}
\begin{document}
"""


def main():
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    caps, body = [], []
    for name, fn in FIGS:
        try:
            fig = fn()
        except (FileNotFoundError, OSError) as e:
            print(f"  skip fig_{name}: {e}")
            continue
        tikz = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", fig, re.S).group(0)
        cap = re.search(r"\\caption\{(.*)\}\s*\\label", fig, re.S)
        with open(os.path.join(out, f"fig_{name}.tex"), "w", encoding="utf-8", newline="\n") as f:
            f.write(tikz + "\n")
        caps.append(f"fig_{name}: {cap.group(1) if cap else ''}")
        body.append(r"\begin{center}\input{out/fig_%s.tex}\end{center}" % name
                    + "\n\\noindent\\textbf{fig\\_%s}\\par\\bigskip\n" % name)
        print("  -> figures/out/fig_%s.tex" % name)
    with open(os.path.join(out, "captions.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(caps) + "\n")
    with open(os.path.join(HERE, "preview.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(PREAMBLE.replace(r"\renewcommand{\E}", r"\providecommand{\E}{}\renewcommand{\E}")
                + "\n".join(body) + "\\end{document}\n")
    print("  -> figures/preview.tex")


if __name__ == "__main__":
    main()
