# -*- coding: utf-8 -*-
r"""Figures 1, 2 and 7 of the paper, drawn in TikZ from the model itself.

Every curve is computed here -- BS densities, n-fold convolutions, simulated
paths, the stored first-passage and maintenance results -- and written as TikZ
coordinates.  Nothing is sketched; the paths are seeded, so an unchanged input
gives an unchanged figure.  Use export_figures.py to write all figures.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import invgauss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TPL = os.path.join(HERE, "template.tex")
sys.path.insert(0, os.path.join(ROOT, "code"))
import gbs                                                        # noqa: E402

A0, B0 = 0.4, 1.0                       # reference law of one unit step
M1, V1 = float(gbs.bs_mean(A0, B0)), float(gbs.bs_var(A0, B0))
NMAX = 5 * M1 ** 2 / V1                 # finest admissible grid, one unit horizon


def g(a):
    a = np.asarray(a, float)
    return a * a * (1 + 1.25 * a * a) / (1 + 0.5 * a * a) ** 2


def cons(n):
    a, b = gbs.bs_consistent_params(A0, B0, 1.0 / n)
    return float(a), float(b)


def P(xs, ys):
    return " ".join("(%.3f,%.3f)" % (x, y) for x, y in zip(xs, ys))


def curve(style, xs, ys):
    return r"\draw[%s] plot coordinates {%s};" % (style, P(xs, ys))


COLORS = (r"\definecolor{cons}{rgb}{0.122,0.306,0.475}"
          r"\definecolor{intv}{rgb}{0.753,0.314,0.302}"
          r"\definecolor{ink}{rgb}{0.20,0.20,0.22}"
          r"\definecolor{alt}{rgb}{0.310,0.561,0.247}")
ARROW = r"-{Stealth[length=3.2pt,width=2.6pt]}"


#  Weibull with the same mean and squared CV as BS(A0, B0) per unit time, so the
#  two moment-consistent laws differ only in shape
def g_w(nu):
    from scipy.special import gamma as G
    nu = np.asarray(nu, float)
    return G(1 + 2 / nu) / G(1 + 1 / nu) ** 2 - 1


def weib_shape(cv2):
    from scipy.optimize import brentq
    return brentq(lambda v: float(g_w(v)) - cv2, 0.02, 500.0)


def weib(mean, cv2):
    from scipy.special import gamma as G
    from scipy.stats import weibull_min
    nu = weib_shape(cv2)
    return weibull_min(nu, scale=mean / G(1 + 1 / nu))


# ---------------------------------------------------------------------------
# Figure 1: one increment -> variance budget -> law of the sum
# ---------------------------------------------------------------------------
def fig_mechvar():
    NS = [1, 2, 4, 8]
    SH = {1: 28, 2: 52, 4: 76, 8: 100}
    H = 2.25                                  # panel height, cm
    YB = {"intv": 7.1, "cons": 3.55, "alt": 0.0}   # baseline of each row
    CV1 = float(g(A0))                        # squared CV of one unit step
    C1, W1, X1 = 0.0, 3.55, 3.2               # col 1: x0, width, data range
    C2 = 4.35
    C3, W3, X3 = 8.5, 3.6, 2.6

    # one increment, rescaled by its mean
    y = np.linspace(0.004, X1, 170)
    c1 = {}
    for tag in ("intv", "cons"):
        for n in NS:
            if tag == "intv":
                a, b = A0, B0 / n
            else:
                a, b = cons(n)
            mean = float(gbs.bs_mean(a, b))
            c1[tag, n] = mean * gbs.bs_pdf(mean * y, a, b)
    top1 = max(v.max() for v in c1.values())
    sy1 = H * 0.86 / top1
    for n in NS:                              # Weibull, same moments per step
        w = weib(M1 / n, CV1 * n)
        c1["alt", n] = (M1 / n) * w.pdf((M1 / n) * y)

    # law of X(T) by n-fold convolution on a fine grid
    dx = 0.001
    x = np.arange(dx / 2, 4.0, dx)
    c3 = {}
    for tag in ("intv", "cons"):
        for n in NS:
            a, b = (A0, B0 / n) if tag == "intv" else cons(n)
            p = gbs.bs_pdf(x, a, b) * dx
            L = 1 << int(np.ceil(np.log2(len(x) * n + 1)))
            f = np.fft.irfft(np.fft.rfft(p, L) ** n, L)[: len(x)]
            #  the sum of n cell-centred variables sits n/2 cells to the right
            xs = x + (n - 1) * dx / 2
            c3[tag, n] = (xs, np.clip(f, 0, None) / dx)
    for n in NS:
        #  cell masses from CDF differences: a Weibull density with shape
        #  below one is unbounded at zero
        w = weib(M1 / n, CV1 * n)
        p = np.diff(w.cdf(np.r_[0.0, x + dx / 2]))
        L = 1 << int(np.ceil(np.log2(len(x) * n + 1)))
        f = np.fft.irfft(np.fft.rfft(p, L) ** n, L)[: len(x)]
        c3["alt", n] = (x + (n - 1) * dx / 2, np.clip(f, 0, None) / dx)
    top3 = max(v[1].max() for v in c3.values())
    sy3 = H * 0.86 / top3
    keep = lambda xs: (xs > 0) & (xs < X3)

    L = [r"\begin{figure}[tbp]", r"\centering",
         r"\begin{tikzpicture}[font=\scriptsize,line cap=round,line join=round]",
         COLORS]
    ttl = r"\node[anchor=south west,font=\footnotesize,inner sep=0] at (%.2f,%.2f) {%s};"
    yt = YB["intv"] + H + 0.32
    L += [ttl % (C1, yt, r"(i) one increment, rescaled"),
          ttl % (C2 - 0.15, yt, r"(ii) variance budget"),
          ttl % (C3, yt, r"(iii) law of $X(T)$")]

    for tag, rowname in (("intv", r"interval scaling\\BS"),
                         ("cons", r"moment-consistent\\BS"),
                         ("alt", r"moment-consistent\\Weibull")):
        yb = YB[tag]
        L.append(r"\node[rotate=90,anchor=south,align=center,font=\footnotesize\bfseries,"
                 r"text=%s] at (-0.34,%.2f) {%s};" % (tag, yb + H / 2, rowname))

        # ---- column 1 ----------------------------------------------------
        L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.2f) -- (%.2f,%.2f);"
                 % (C1, yb, C1 + W1, yb))
        for t in range(0, 4):
            xx = C1 + t * W1 / X1
            L.append(r"\draw[ink,line width=0.45pt] (%.3f,%.2f) -- ++(0,-0.07)"
                     r" node[below,inner sep=1.5pt] {$%d$};" % (xx, yb, t))
        L.append(r"\draw[ink!50,densely dotted] (%.3f,%.2f) -- (%.3f,%.2f);"
                 % (C1 + W1 / X1, yb, C1 + W1 / X1, yb + H * 0.95))
        L.append(r"\node[anchor=north east,inner sep=1pt] at (%.2f,%.2f)"
                 r" {$\Delta X/\E[\Delta X]$};" % (C1 + W1, yb - 0.36))
        order = NS if tag != "intv" else [8]
        for n in order:
            xs = C1 + y * W1 / X1
            #  clipped at the panel top: Weibull densities with shape below
            #  one are unbounded at zero
            ys = yb + np.minimum(c1[tag, n] * sy1, H * 0.97)
            if tag != "intv" and n == 1:
                L.append(r"\fill[%s!10] (%.3f,%.3f) -- plot coordinates {%s}"
                         r" -- (%.3f,%.3f) -- cycle;" % (tag, xs[0], yb, P(xs, ys), xs[-1], yb))
            L.append(curve("%s!%d,line width=%.2fpt" % (tag, SH[n], 0.7 + 0.1 * NS.index(n)),
                           xs, ys))
        if tag == "intv":
            L.append(r"\fill[intv!10] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f)"
                     r" -- cycle;" % (C1, yb, P(C1 + y * W1 / X1, yb + c1["intv", 8] * sy1),
                                      C1 + W1, yb))
            L.append(curve("intv,line width=1.0pt", C1 + y * W1 / X1, yb + c1["intv", 8] * sy1))
            L.append(r"\node[anchor=west,align=left,inner sep=0] at (%.2f,%.2f)"
                     r" {one shape on\\[-1pt]every grid};"
                     % (C1 + 1.75, yb + H * 0.80))
        elif tag == "cons":
            L.append(r"\node[anchor=west,align=left,inner sep=0] at (%.2f,%.2f)"
                     r" {$\alpha_s$ grows,\\[-1pt]spread $\propto\sqrt{n}$};"
                     % (C1 + 1.75, yb + H * 0.80))
        else:
            L.append(r"\node[anchor=west,align=left,inner sep=0] at (%.2f,%.2f)"
                     r" {$\nu_s$ falls,\\[-1pt]spread $\propto\sqrt{n}$};"
                     % (C1 + 1.75, yb + H * 0.80))

        # ---- column 2 ----------------------------------------------------
        Hs = H * 0.74
        ax0 = C2 + 0.05
        L.append(r"\draw[ink,line width=0.45pt,%s] (%.2f,%.2f) -- (%.2f,%.2f);"
                 % (ARROW, ax0, yb, ax0, yb + H * 0.98))
        L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.2f) -- (%.2f,%.2f);"
                 % (ax0, yb, ax0 + 3.55, yb))
        L.append(r"\draw[ink!45,densely dashed] (%.2f,%.3f) -- (%.2f,%.3f);"
                 % (ax0, yb + Hs, ax0 + 3.55, yb + Hs))
        L.append(r"\node[left,inner sep=1.5pt] at (%.2f,%.3f) {$V_1$};" % (ax0, yb + Hs))
        w, gap = 0.6, 0.24
        for k, n in enumerate(NS):
            x0 = ax0 + 0.3 + k * (w + gap)
            h = Hs / n ** 2 if tag == "intv" else Hs / n
            for i in range(n):
                L.append(r"\filldraw[fill=%s!%d,draw=white,line width=0.35pt]"
                         r" (%.3f,%.4f) rectangle ++(%.2f,%.4f);"
                         % (tag, SH[n], x0, yb + i * h, w, h))
            if tag == "intv" and n > 1:
                lab = r"V_1/%d" % n
                L.append(r"\node[above,inner sep=1.2pt] at (%.3f,%.3f) {$%s$};"
                         % (x0 + w / 2, yb + n * h, lab))
            L.append(r"\draw[ink,line width=0.4pt] (%.3f,%.2f) -- ++(%.2f,0);"
                     % (x0, yb - 0.17, w))
            for j in range(n + 1):
                L.append(r"\draw[ink,line width=0.4pt] (%.3f,%.2f) -- ++(0,0.08);"
                         % (x0 + w * j / n, yb - 0.21))
            L.append(r"\node[below,inner sep=1pt] at (%.3f,%.2f) {$n=%d$};"
                     % (x0 + w / 2, yb - 0.21, n))
        fml = (r"$n\times V_1/n^{2}=V_1/n$" if tag == "intv"
               else r"$n\times V_1/n=V_1$")
        L.append(r"\node[anchor=north east,inner sep=0,text=%s] at (%.2f,%.2f) {%s};"
                 % (tag, ax0 + 3.55, yb + H * 0.98, fml))

        # ---- column 3 ----------------------------------------------------
        L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.2f) -- (%.2f,%.2f);"
                 % (C3, yb, C3 + W3, yb))
        for t in (0, 1, 2):
            xx = C3 + t * W3 / X3
            L.append(r"\draw[ink,line width=0.45pt] (%.3f,%.2f) -- ++(0,-0.07)"
                     r" node[below,inner sep=1.5pt] {$%d$};" % (xx, yb, t))
        L.append(r"\draw[ink!50,densely dotted] (%.3f,%.2f) -- (%.3f,%.2f);"
                 % (C3 + M1 * W3 / X3, yb, C3 + M1 * W3 / X3, yb + H * 0.95))
        L.append(r"\node[anchor=north east,inner sep=1pt] at (%.2f,%.2f) {$X(T)$};"
                 % (C3 + W3, yb - 0.36))
        for n in NS:
            xs, f = c3[tag, n]
            k = keep(xs)
            idx = np.linspace(0, k.sum() - 1, 180).astype(int)
            xx, ff = xs[k][idx], f[k][idx]
            L.append(curve("%s!%d,line width=%.2fpt" % (tag, SH[n], 0.6 + 0.12 * NS.index(n)),
                           C3 + xx * W3 / X3, yb + ff * sy3))
        if tag == "intv":
            L.append(r"\node[anchor=west,align=left,inner sep=0] at (%.2f,%.2f)"
                     r" {narrows:\\[-1pt]sd $\propto 1/\sqrt{n}$};"
                     % (C3 + 2.2, yb + H * 0.80))
        else:
            L.append(r"\node[anchor=west,align=left,inner sep=0] at (%.2f,%.2f)"
                     r" {same mean\\[-1pt]and variance};"
                     % (C3 + 2.2, yb + H * 0.80))
        if tag == "alt":
            #  the BS consistent law of X(T) at n = 1, for comparison of shape
            xs, f = c3["cons", 1]
            k = keep(xs)
            idx = np.linspace(0, k.sum() - 1, 180).astype(int)
            L.append(curve("cons!70,densely dashed,line width=0.6pt",
                           C3 + xs[k][idx] * W3 / X3, yb + f[k][idx] * sy3))
            L.append(r"\node[anchor=west,inner sep=0,text=cons!80] at (%.2f,%.2f)"
                     r" {dashed: BS};" % (C3 + 2.2, yb + H * 0.45))

        # ---- links between columns ----------------------------------------
        for xa, xb in ((C1 + W1 + 0.1, C2 - 0.42), (ax0 + 3.6, C3 - 0.12)):
            L.append(r"\draw[ink!55,line width=0.6pt,%s] (%.2f,%.2f) -- (%.2f,%.2f);"
                     % (ARROW, xa, yb + H * 0.42, xb, yb + H * 0.42))
        L.append(r"\node[ink!70,above,inner sep=1pt] at (%.2f,%.2f) {$\times n$};"
                 % ((C1 + W1 + C2 - 0.32) / 2, yb + H * 0.42))
        L.append(r"\node[ink!70,above,inner sep=1pt] at (%.2f,%.2f) {$\ast n$};"
                 % ((ax0 + 3.6 + C3 - 0.12) / 2, yb + H * 0.42))

    # legend: shade encodes the grid
    ly = -0.95
    L.append(r"\node[anchor=east,inner sep=1pt] at (4.6,%.2f) {darker = finer grid:};" % ly)
    for k, n in enumerate(NS):
        xx = 4.75 + k * 1.45
        for j, col in enumerate(("intv", "cons", "alt")):
            L.append(r"\fill[%s!%d] (%.2f,%.2f) rectangle ++(0.16,0.14);"
                     % (col, SH[n], xx + 0.16 * j, ly - 0.07))
        L.append(r"\node[anchor=west,inner sep=1pt] at (%.2f,%.2f) {$n=%d$};" % (xx + 0.5, ly, n))
    L += [r"\end{tikzpicture}",
          r"\caption{The mechanism of \Cref{prop:griddep} over a unit horizon split"
          r" into $n$ steps, for $\BS(0.4,1)$ and a Weibull law with the same mean and"
          r" variance. Rows: interval scaling (BS); moment-consistent BS; moment-consistent"
          r" Weibull. (i)~One increment in units of its mean. (ii)~Variance budget, one"
          r" block per increment. (iii)~Exact law of $X(T)$ by numerical convolution.}",
          r"\label{fig:mechvar}", r"\end{figure}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Figure 2: the root of g(alpha) = r, the finest grid, and the classification
# ---------------------------------------------------------------------------
def fig_mechroot():
    NS = [1, 2, 4, 8, 16]
    SH = {1: 30, 2: 48, 4: 66, 8: 84, 16: 100}
    r1 = float(g(A0))
    # (i) log-log axes
    la0, la1, lg0, lg1 = np.log10(0.2), np.log10(20), np.log10(0.04), np.log10(20)
    WX, WY = 4.6, 2.75
    X = lambda a: (np.log10(a) - la0) / (la1 - la0) * WX
    Y = lambda v: (np.log10(v) - lg0) / (lg1 - lg0) * WY

    L = [r"\begin{figure}[tbp]", r"\centering",
         r"\begin{tikzpicture}[font=\scriptsize,line cap=round,line join=round]",
         COLORS]
    L.append(r"\begin{scope}")
    L.append(r"\fill[intv!9] (0,%.3f) rectangle (%.2f,%.3f);" % (Y(5), WX, WY))
    L.append(r"\draw[intv!70,line width=0.6pt] (0,%.3f) -- (%.2f,%.3f);" % (Y(5), WX, Y(5)))
    L.append(r"\node[anchor=north east,text=intv!85!black,inner sep=2pt] at (%.2f,%.3f)"
             r" {no BS root above $5$};" % (WX, WY))
    L.append(r"\draw[ink,line width=0.45pt,%s] (0,0) -- (%.2f,0);" % (ARROW, WX + 0.25))
    L.append(r"\draw[ink,line width=0.45pt,%s] (0,0) -- (0,%.2f);" % (ARROW, WY + 0.2))
    for a in (0.2, 0.5, 1, 2, 5, 10, 20):
        L.append(r"\draw[ink,line width=0.45pt] (%.3f,0) -- ++(0,-0.07)"
                 r" node[below,inner sep=1.5pt] {$%s$};" % (X(a), ("%g" % a)))
    L.append(r"\node[anchor=north east,inner sep=1pt] at (%.2f,-0.32)"
             r" {shape $\alpha$ (BS) or $\nu$ (Weibull), log scale};" % WX)
    for v in (0.1, 1, 5):
        L.append(r"\draw[ink,line width=0.45pt] (0,%.3f) -- ++(-0.07,0);" % Y(v))
    L.append(r"\node[anchor=south west,inner sep=1pt] at (0.05,%.2f) {$g$, log scale};" % (WY + 0.12))
    aa = np.logspace(la0, la1, 160)
    L.append(r"\draw[ink!35,densely dashed,line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);"
             % (X(0.2), Y(0.04), X(np.sqrt(20.0)), WY))
    L.append(curve("cons,line width=1.1pt", X(aa), Y(g(aa))))
    #  Weibull with the same moments: g_W falls with the shape and is unbounded
    vv = np.logspace(la0, la1, 220)
    gv = g_w(vv)
    ok = (gv <= 20.0) & (gv >= 0.04)
    L.append(curve("alt,line width=1.1pt", X(vv[ok]), Y(gv[ok])))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=cons] at (%.3f,%.3f) {BS};"
             % (X(9.0), Y(float(g(9.0))) - 0.36))
    L.append(r"\node[anchor=west,inner sep=1.5pt,text=alt!80!black] at (%.3f,%.3f) {Weibull};"
             % (X(3.2), Y(float(g_w(3.2)))))
    from scipy.optimize import brentq
    for n in NS:
        r = n * r1
        a = brentq(lambda t: float(g(t)) - r, 1e-6, 1e6)
        nu = weib_shape(r)
        L.append(r"\draw[ink!%d,densely dotted,line width=0.6pt] (0,%.3f) -- (%.3f,%.3f);"
                 % (SH[n] // 2 + 20, Y(r), max(X(a), X(nu)), Y(r)))
        L.append(r"\draw[cons!%d,densely dotted,line width=0.7pt] (%.3f,%.3f) -- (%.3f,0);"
                 % (SH[n], X(a), Y(r), X(a)))
        L.append(r"\fill[cons!%d] (%.3f,%.3f) circle (1.5pt);" % (SH[n], X(a), Y(r)))
        L.append(r"\fill[alt!%d] (%.3f,%.3f) circle (1.5pt);" % (SH[n], X(nu), Y(r)))
        L.append(r"\node[left,inner sep=1.5pt,text=ink] at (0,%.3f) {$n=%d$};" % (Y(r), n))
    nu32 = weib_shape(32 * r1)
    L.append(r"\draw[intv,densely dashed,line width=0.7pt] (0,%.3f) -- (%.2f,%.3f);"
             % (Y(32 * r1), WX * 0.55, Y(32 * r1)))
    L.append(r"\filldraw[fill=alt,draw=white,line width=0.4pt] (%.3f,%.3f) circle (1.8pt);"
             % (X(nu32), Y(32 * r1)))
    L.append(r"\node[left,inner sep=1.5pt,text=intv] at (0,%.3f) {$n=32$};" % Y(32 * r1))
    L.append(r"\node[anchor=south west,font=\footnotesize,inner sep=0] at (-0.85,%.2f)"
             r" {(i) the shape is the root of $g=r$};" % (WY + 0.55))
    L.append(r"\end{scope}")

    # (ii) alpha_s and the scale ratio against n
    x0 = 6.55
    WX2, WY2, AMAX = 4.3, 2.75, 6.0
    U = lambda n: x0 + np.log2(n) / 5.0 * WX2
    V = lambda a: np.minimum(a, AMAX) / AMAX * WY2
    L.append(r"\draw[ink,line width=0.45pt] (%.2f,0) -- (%.2f,0);" % (x0, x0 + WX2))
    L.append(r"\draw[ink,line width=0.45pt,%s] (%.2f,0) -- (%.2f,%.2f);" % (ARROW, x0, x0, WY2 + 0.2))
    L.append(r"\draw[ink!60,line width=0.45pt] (%.2f,0) -- (%.2f,%.2f);" % (x0 + WX2, x0 + WX2, WY2))
    for n in (1, 2, 4, 8, 16, 32):
        L.append(r"\draw[ink,line width=0.45pt] (%.3f,0) -- ++(0,-0.07) node[below,inner sep=1.5pt] {$%d$};"
                 % (U(n), n))
    L.append(r"\node[anchor=north east,inner sep=1pt] at (%.2f,-0.32) {steps $n$ over the horizon, log scale};"
             % (x0 + WX2))
    for a in (0, 2, 4, 6):
        L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.3f) -- ++(-0.07,0) node[left,inner sep=1.5pt] {$%d$};"
                 % (x0, V(a), a))
    for q in (1,):
        L.append(r"\draw[ink!60,line width=0.45pt] (%.2f,%.3f) -- ++(0.07,0) node[right,inner sep=1.5pt,text=ink!70] {$%g$};"
                 % (x0 + WX2, q * WY2, q))
    L.append(r"\node[anchor=south west,inner sep=1pt] at (%.2f,%.2f)"
             r" {\textcolor{cons}{$\alpha_s$}, \textcolor{alt!80!black}{$\nu_s$}};"
             % (x0 + 0.05, WY2 + 0.12))
    L.append(r"\node[rotate=90,anchor=north,inner sep=1pt,text=ink!70] at (%.2f,%.2f) {$\beta_s/(s\beta)$};"
             % (x0 + WX2 + 0.42, WY2 * 0.5))
    L.append(r"\fill[intv!9] (%.3f,0) rectangle (%.2f,%.2f);" % (U(NMAX), x0 + WX2, WY2))
    L.append(r"\draw[intv!70,line width=0.6pt] (%.3f,0) -- (%.3f,%.2f);" % (U(NMAX), U(NMAX), WY2))
    L.append(r"\node[anchor=south east,inner sep=1.5pt,text=intv!85!black] at (%.3f,%.3f)"
             r" {$n_{\max}$};" % (U(NMAX) - 0.12, WY2 + 0.05))
    nn = np.exp(np.linspace(0, np.log(NMAX * 0.9995), 220))
    als = np.array([cons(n)[0] for n in nn])
    bss = np.array([cons(n)[1] for n in nn])
    rho = bss / (B0 / nn)
    k = als <= AMAX
    L.append(curve("cons,line width=1.1pt", U(nn[k]), V(als[k])))
    L.append(r"\draw[cons,line width=1.1pt,%s] (%.3f,%.3f) -- ++(0,0.12);" % (ARROW, U(nn[k][-1]), V(als[k][-1])))
    L.append(curve("ink!65,densely dashed,line width=0.8pt", U(nn), rho * WY2))
    for n in NS:
        a, _ = cons(n)
        L.append(r"\fill[cons!%d] (%.3f,%.3f) circle (1.5pt);" % (SH[n], U(n), V(a)))
    #  Weibull shape along the grid: it falls smoothly and meets no wall
    nw = np.exp(np.linspace(0, np.log(32.0), 200))
    L.append(curve("alt,line width=1.1pt", U(nw), V(np.array([weib_shape(n * r1) for n in nw]))))
    for n in NS + [32]:
        L.append(r"\fill[alt!%d] (%.3f,%.3f) circle (1.5pt);"
                 % (SH.get(n, 100), U(n), V(weib_shape(n * r1))))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=alt!80!black] at (%.3f,%.3f)"
             r" {$\nu_s$: no wall};" % (U(1.15), V(weib_shape(1.15 * r1))))
    L.append(r"\node[anchor=south west,font=\footnotesize,inner sep=0] at (%.2f,%.2f)"
             r" {(ii) the increment law along the grid};" % (x0 - 0.3, WY2 + 0.55))

    L += [r"\end{tikzpicture}",
          r"\caption{How the shape of a consistent increment is set, for $\BS(0.4,1)$"
          r" (blue) and a Weibull law with the same mean and variance (green), over a"
          r" unit horizon. (i)~The shape is the root of $g=r$, with $r\propto n$. BS has"
          r" no root above $g=5$; Weibull has a root on every grid. (ii)~$\alpha_s$,"
          r" $\nu_s$ and the BS scale ratio $\beta_s/(s\beta)$ against $n$; only BS stops"
          r" at $n_{\max}$ (\Cref{rem:finegrid}).}",
          r"\label{fig:mechroot}", r"\end{figure}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Figure 3: from paths to first passage to a replacement plan
# ---------------------------------------------------------------------------
def fig_mechrisk():
    G = json.load(open(os.path.join(ROOT, "results/theory/grid_consistency.json"), encoding="utf-8"))
    Mx = json.load(open(os.path.join(ROOT, "results/theory/maintenance_example.json"), encoding="utf-8"))
    D = G["D"]
    NST = 16
    fp = {k: [r for r in G["first_passage"][k] if r["steps_per_unit"] == NST][0]
          for k in ("naive", "consistent")}

    def ig(m, s):
        lam = m ** 3 / s ** 2
        return invgauss(mu=m / lam, scale=lam)
    TRUE, NAIVE = ig(fp["consistent"]["mean"], fp["consistent"]["sd"]), \
        ig(fp["naive"]["mean"], fp["naive"]["sd"])
    row = [r for r in Mx["rows"] if r["steps_per_unit"] == NST][0]
    qn, qc = row["q_naive"], row["q_consistent"]
    risk = row["true_risk_at_naive_plan"]

    T0, T1, SX = 0.0, 16.5, 0.45                   # time axis, cm per unit
    TX = lambda t: (np.asarray(t) - T0) * SX
    YP0, SP = 2.55, 2.3 / 13.0                     # path tier
    YD0, HD = 0.0, 2.05                            # density tier
    dmax = TRUE.pdf(TRUE.mean())
    SD = HD * 0.88 / dmax

    L = [r"\begin{figure}[tbp]", r"\centering",
         r"\begin{tikzpicture}[font=\scriptsize,line cap=round,line join=round]",
         COLORS]
    # path tier ---------------------------------------------------------------
    rng = np.random.default_rng(7)
    a_s, b_s = cons(NST)
    taus = []
    for _ in range(26):
        t, x, ts, xs = 0.0, 0.0, [0.0], [0.0]
        step = 0
        while True:
            dx = float(gbs.bs_rvs(a_s, b_s, rng=rng))
            if x + dx >= D:
                t_cross = t + (D - x) / dx / NST
                ts.append(t_cross); xs.append(D); break
            t += 1.0 / NST; x += dx; step += 1
            if step % 4 == 0:
                ts.append(t); xs.append(x)
        taus.append(ts[-1])
        L.append(curve("cons!38,line width=0.35pt", TX(ts), YP0 + np.array(xs) * SP))
    for tau in taus:
        L.append(r"\draw[cons!25,line width=0.3pt] (%.3f,%.3f) -- (%.3f,%.3f);"
                 % (TX(tau), YP0 + D * SP, TX(tau), YP0))
        L.append(r"\fill[cons] (%.3f,%.3f) circle (0.9pt);" % (TX(tau), YP0 + D * SP))
    L.append(r"\draw[ink,line width=0.45pt,%s] (0,%.2f) -- (0,%.2f);" % (ARROW, YP0, YP0 + 13.6 * SP))
    L.append(r"\draw[ink,line width=0.45pt] (0,%.2f) -- (%.2f,%.2f);" % (YP0, TX(T1), YP0))
    L.append(r"\draw[intv!80,line width=0.7pt] (0,%.3f) -- (%.2f,%.3f) node[right,inner sep=1.5pt,text=intv] {$D$};"
             % (YP0 + D * SP, TX(T1), YP0 + D * SP))
    L.append(r"\node[anchor=south west,inner sep=1pt] at (0.05,%.2f) {degradation $X(t)$};" % (YP0 + 13.6 * SP))
    for k in range(int(T1 * NST) + 1):
        L.append(r"\draw[ink!30,line width=0.2pt] (%.3f,%.2f) -- ++(0,0.05);" % (TX(k / NST), YP0))
    # density tier ----------------------------------------------------------
    tt = np.linspace(6.0, T1, 400)
    ft = TRUE.pdf(tt)
    fn = NAIVE.pdf(tt) / 4.0
    for lo, hi, fill in ((6.0, qc, "cons!55"), (qc, qn, "intv!45")):
        m = (tt >= lo) & (tt <= hi)
        xs = np.r_[lo, tt[m], hi]
        ys = np.r_[TRUE.pdf(lo), ft[m], TRUE.pdf(hi)]
        L.append(r"\fill[%s] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f) -- cycle;"
                 % (fill, TX(lo), YD0, P(TX(xs), YD0 + ys * SD), TX(hi), YD0))
    L.append(r"\fill[cons!10] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f) -- cycle;"
             % (TX(qn), YD0, P(TX(tt[tt >= qn]), YD0 + ft[tt >= qn] * SD), TX(T1), YD0))
    L.append(curve("cons,line width=1.0pt", TX(tt), YD0 + ft * SD))
    k = fn * SD > 0.004
    L.append(curve("intv,densely dashed,line width=0.9pt", TX(tt[k]), YD0 + fn[k] * SD))
    L.append(r"\draw[ink,line width=0.45pt,%s] (0,%.2f) -- (%.2f,%.2f);" % (ARROW, YD0, TX(T1) + 0.25, YD0))
    for t in range(0, 17, 2):
        L.append(r"\draw[ink,line width=0.45pt] (%.3f,%.2f) -- ++(0,-0.07) node[below,inner sep=1.5pt] {$%d$};"
                 % (TX(t), YD0, t))
    L.append(r"\node[anchor=north east,inner sep=1pt] at (%.2f,%.2f) {time};" % (TX(T1) + 0.25, YD0 - 0.3))
    for t, col, lab, yy in ((qc, "cons", r"consistent plan", 1.02), (qn, "intv", r"interval-scaled plan", 1.02)):
        L.append(r"\draw[%s,line width=0.8pt] (%.3f,%.2f) -- (%.3f,%.2f);" % (col, TX(t), YD0, TX(t), YD0 + HD * yy))
    L.append(r"\node[anchor=south east,inner sep=1pt,text=cons] at (%.3f,%.2f) {consistent plan};"
             % (TX(qc) - 0.03, YD0 + HD * 0.98))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=intv] at (%.3f,%.2f) {interval-scaled plan};"
             % (TX(qn) + 0.03, YD0 + HD * 0.98))
    L.append(r"\node[anchor=east,align=right,inner sep=1pt,text=cons!80!black] at (%.3f,%.2f)"
             r" {$5\,\%%$ planned};" % (TX(6.6), YD0 + 0.3))
    L.append(r"\draw[ink!60,line width=0.4pt] (%.3f,%.2f) -- (%.3f,%.2f);"
             % (TX(6.65), YD0 + 0.3, TX(qc - 0.35), YD0 + 0.07))
    L.append(r"\node[anchor=east,align=right,inner sep=1pt,text=intv!85!black] at (%.3f,%.2f)"
             r" {$+%.1f\,\%%$ unplanned failures,\\[-1pt]$%.1f\,\%%$ in all};"
             % (TX(6.6), YD0 + 0.95, 100 * (risk - Mx["eta"]), 100 * risk))
    L.append(r"\draw[ink!60,line width=0.4pt] (%.3f,%.2f) -- (%.3f,%.2f);"
             % (TX(6.65), YD0 + 0.95, TX((qc + qn) / 2), YD0 + 0.6))
    L.append(r"\draw[cons,line width=1.0pt] (%.2f,%.2f) -- ++(0.45,0) node[right,inner sep=1.5pt,text=ink] {first-passage law};"
             % (TX(12.6), YD0 + HD * 0.86))
    L.append(r"\draw[intv,densely dashed,line width=0.9pt] (%.2f,%.2f) -- ++(0.45,0) node[right,inner sep=1.5pt,text=ink] {interval-scaled, $\times\frac14$};"
             % (TX(12.6), YD0 + HD * 0.68))
    L.append(r"\node[anchor=south west,font=\footnotesize,inner sep=0] at (-0.6,%.2f)"
             r" {(a) paths, first passage and the two plans};" % (YP0 + 13.6 * SP + 0.45))

    # (b) plans and risk against inspection rate --------------------------------
    xb, WB = 9.25, 3.4
    U = lambda k: xb + k / 4.0 * WB
    rows = Mx["rows"]
    lk = [np.log2(r["steps_per_unit"]) for r in rows]
    # top: plan times
    y1, h1, t_lo, t_hi = 2.75, 1.75, 9.0, 10.7
    Yt = lambda t: y1 + (t - t_lo) / (t_hi - t_lo) * h1
    qn_ = [r["q_naive"] for r in rows]; qc_ = [r["q_consistent"] for r in rows]
    L.append(r"\fill[intv!12] plot coordinates {%s} -- plot coordinates {%s} -- cycle;"
             % (P([U(k) for k in lk], [Yt(q) for q in qn_]),
                P([U(k) for k in lk[::-1]], [Yt(q) for q in qc_[::-1]])))
    L.append(r"\draw[ink,line width=0.45pt,%s] (%.2f,%.2f) -- (%.2f,%.2f);" % (ARROW, xb, y1, xb, y1 + h1 + 0.2))
    L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (xb, y1, xb + WB + 0.1, y1))
    for t in (9, 10):
        L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.3f) -- ++(-0.07,0) node[left,inner sep=1.5pt] {$%d$};" % (xb, Yt(t), t))
    L.append(curve("intv,line width=0.9pt,mark=*,mark size=1.3pt", [U(k) for k in lk], [Yt(q) for q in qn_]))
    L.append(curve("cons,line width=0.9pt,mark=*,mark size=1.3pt", [U(k) for k in lk], [Yt(q) for q in qc_]))
    L.append(r"\node[anchor=south west,inner sep=1pt] at (%.2f,%.2f) {replacement time};" % (xb + 0.05, y1 + h1 + 0.1))
    L.append(r"\node[inner sep=1pt,text=intv!85!black] at (%.2f,%.3f) {postponed};"
             % (U(3.05), Yt(9.65)))
    # bottom: carried risk
    y2, h2, rmax = 0.0, 1.95, 36.0
    Yr = lambda p: y2 + p / rmax * h2
    rk = [100 * r["true_risk_at_naive_plan"] for r in rows]
    L.append(r"\fill[intv!14] plot coordinates {%s} -- (%.3f,%.3f) -- (%.3f,%.3f) -- cycle;"
             % (P([U(k) for k in lk], [Yr(p) for p in rk]), U(lk[-1]), Yr(5), U(lk[0]), Yr(5)))
    L.append(r"\draw[ink,line width=0.45pt,%s] (%.2f,%.2f) -- (%.2f,%.2f);" % (ARROW, xb, y2, xb, y2 + h2 + 0.2))
    L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (xb, y2, xb + WB + 0.1, y2))
    for p in (5, 20, 35):
        L.append(r"\draw[ink,line width=0.45pt] (%.2f,%.3f) -- ++(-0.07,0) node[left,inner sep=1.5pt] {$%d$};" % (xb, Yr(p), p))
    L.append(r"\draw[ink!45,densely dashed] (%.2f,%.3f) -- (%.2f,%.3f);" % (xb, Yr(5), xb + WB + 0.1, Yr(5)))
    L.append(curve("intv,line width=0.9pt,mark=*,mark size=1.3pt", [U(k) for k in lk], [Yr(p) for p in rk]))
    L.append(r"\node[anchor=south west,inner sep=1pt] at (%.2f,%.2f) {failure risk carried (\%%)};" % (xb + 0.05, y2 + h2 + 0.1))
    L.append(r"\node[anchor=north,inner sep=1pt,text=intv!85!black] at (%.2f,%.3f) {excess};" % (U(3.0), Yr(16)))
    for k, r in zip(lk, rows):
        L.append(r"\draw[ink,line width=0.45pt] (%.3f,%.2f) -- ++(0,-0.07) node[below,inner sep=1.5pt] {$%d$};"
                 % (U(k), y2, r["steps_per_unit"]))
    L.append(r"\node[anchor=north east,inner sep=1pt] at (%.2f,%.2f) {inspections per unit time};" % (xb + WB + 0.1, y2 - 0.3))
    L.append(r"\node[anchor=south west,font=\footnotesize,inner sep=0] at (%.2f,%.2f) {(b) cost of monitoring more};"
             % (xb - 0.75, YP0 + 13.6 * SP + 0.45))
    L += [r"\end{tikzpicture}",
          r"\caption{Replacement at the $5\,\%$ quantile on the first-passage study of"
          r" \Cref{tab:grid}. (a)~Consistent paths at sixteen steps per unit time, their"
          r" first-passage law and the two plans. The interval-scaled law is drawn at a"
          r" quarter of its true height, which is four times that of the consistent"
          r" law. (b)~Plan time and carried risk against inspection rate.}",
          r"\label{fig:mechrisk}", r"\end{figure}"]
    return "\n".join(L)


def splice(src, label, block):
    i = src.index(r"\label{%s}" % label)
    a = src.rindex(r"\begin{figure}", 0, i)
    b = src.index(r"\end{figure}", i) + len(r"\end{figure}")
    return src[:a] + block + src[b:]


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import tikz_data_figs as TD
    s = open(TPL, encoding="utf-8").read()
    figs = (("fig:mechvar", fig_mechvar), ("fig:mechroot", fig_mechroot),
            ("fig:mechrisk", fig_mechrisk), ("fig:lamperti", TD.fig_lamperti),
            ("fig:distances", TD.fig_distances), ("fig:grid", TD.fig_grid),
            ("fig:realdata", TD.fig_realdata),
            ("fig:application", TD.fig_application))
    for lab, fn in figs:
        s = splice(s, lab, fn())
    open(TPL, "w", encoding="utf-8", newline="\n").write(s)
    print("spliced %d TikZ figures; n_max = %.2f" % (len(figs), NMAX))
