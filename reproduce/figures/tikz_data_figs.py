# -*- coding: utf-8 -*-
r"""Figures 3--6 and 8 of the paper in TikZ: the Lamperti map, the two
distances, grid consistency, the real data and the end-to-end application.
Every curve is computed here from the model, from results/, or from the
published data sets in data_external/smrd.  Use export_figures.py.
"""
import json
import os

import numpy as np
from scipy.stats import invgauss

import make_tikz_mech as M

ROOT, gbs, P, curve, ARROW = M.ROOT, M.gbs, M.P, M.curve, M.ARROW
COLORS = M.COLORS + r"\definecolor{acc}{rgb}{0.878,0.510,0.078}"
LW = "line width=0.45pt"


def head(L):
    L += [r"\begin{figure}[tbp]", r"\centering",
          r"\begin{tikzpicture}[font=\scriptsize,line cap=round,line join=round]",
          COLORS]


def tail(L, caption, label):
    L += [r"\end{tikzpicture}", r"\caption{" + caption + "}",
          r"\label{%s}" % label, r"\end{figure}"]
    return "\n".join(L)


def frame(L, x0, y0, w, h, xt=(), yt=(), xlab=None, ylab=None, grid=True,
          ylab_dx=0.55, xlab_dy=0.40, right=False):
    """L-shaped axes; xt / yt are (position in cm from the origin, label)."""
    if grid:
        for p, _ in xt:
            L.append(r"\draw[ink!9,line width=0.3pt] (%.3f,%.3f) -- ++(0,%.3f);" % (x0 + p, y0, h))
        for p, _ in yt:
            L.append(r"\draw[ink!9,line width=0.3pt] (%.3f,%.3f) -- ++(%.3f,0);" % (x0, y0 + p, w))
    L.append(r"\draw[ink,%s] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f);"
             % (LW, x0, y0 + h, x0, y0, x0 + w, y0))
    for p, lab in xt:
        L.append(r"\draw[ink,%s] (%.3f,%.3f) -- ++(0,-0.06) node[below,inner sep=1.4pt] {%s};"
                 % (LW, x0 + p, y0, lab))
    for p, lab in yt:
        L.append(r"\draw[ink,%s] (%.3f,%.3f) -- ++(-0.06,0) node[left,inner sep=1.4pt] {%s};"
                 % (LW, x0, y0 + p, lab))
    if xlab:
        L.append(r"\node[anchor=north,inner sep=1pt] at (%.3f,%.3f) {%s};" % (x0 + w / 2, y0 - xlab_dy, xlab))
    if ylab:
        L.append(r"\node[rotate=90,anchor=south,inner sep=1pt] at (%.3f,%.3f) {%s};"
                 % (x0 - ylab_dx, y0 + h / 2, ylab))


def title(L, x, y, text):
    L.append(r"\node[anchor=south west,font=\footnotesize,inner sep=0] at (%.3f,%.3f) {%s};" % (x, y, text))


def mark(L, x, y, kind, col, size=0.055):
    if kind == "o":
        L.append(r"\filldraw[fill=%s,draw=white,line width=0.3pt] (%.3f,%.3f) circle (%.3f);" % (col, x, y, size))
    elif kind == "s":
        L.append(r"\filldraw[fill=%s,draw=white,line width=0.3pt] (%.3f,%.3f) rectangle ++(%.3f,%.3f);"
                 % (col, x - size, y - size, 2 * size, 2 * size))
    elif kind == "t":
        L.append(r"\filldraw[fill=%s,draw=white,line width=0.3pt] (%.3f,%.3f) -- ++(%.3f,%.3f) -- ++(%.3f,0) -- cycle;"
                 % (col, x, y + 1.2 * size, -1.1 * size, -2 * size, 2.2 * size))
    elif kind == "oo":
        L.append(r"\filldraw[fill=white,draw=%s,line width=0.6pt] (%.3f,%.3f) circle (%.3f);" % (col, x, y, size))


def bs_sample(a, b, rng):
    z = rng.standard_normal(np.shape(a))
    return b / 4 * (a * z + np.sqrt(4 + a * a * z * z)) ** 2


# ---------------------------------------------------------------------------
# Figure 3: the Lamperti map
# ---------------------------------------------------------------------------
GAM, AL, BE, XR, DD = 1.5, 0.3, 0.02, 1.0, 3.0


def q(x):
    return 1 + np.asarray(x) / XR


def psi(x, g=GAM):
    return XR / (1 - g) * (q(x) ** (1 - g) - 1) if g != 1 else XR * np.log(q(x))


def fig_lamperti():
    mu, s2 = float(gbs.bs_mean(AL, BE)), float(gbs.bs_var(AL, BE))
    dZ, dV = float(psi(DD)), float(psi(DD, 2 * GAM))
    kap = s2 / (2 * mu) * (1 - q(DD) ** -GAM) + GAM * mu / 2 * np.log(q(DD))
    rng = np.random.default_rng(11)
    N, K = 20000, 75
    X = np.zeros((K + 1, N))
    tau = np.full(N, np.nan)
    for k in range(1, K + 1):
        s = q(X[k - 1]) ** GAM
        a, b = gbs.bs_consistent_params(AL, BE, s)
        dx = bs_sample(np.asarray(a), np.asarray(b), rng)
        X[k] = X[k - 1] + dx
        cr = np.isnan(tau) & (X[k] >= DD)
        tau[cr] = k - 1 + (DD - X[k - 1, cr]) / dx[cr]
    assert not np.isnan(tau).any()
    Z = psi(X)

    L = []
    head(L)
    H, Y1 = 2.55, 3.0
    KX = 62.0
    # (a) X-space ---------------------------------------------------------------
    xa, wa = 0.0, 3.75
    sxk, sxx = wa / KX, H / 3.3
    frame(L, xa, Y1, wa, H, [(k * sxk, "$%d$" % k) for k in (0, 20, 40, 60)],
          [(v * sxx, "$%g$" % v) for v in (0, 1, 2, 3)], "inspection index $k$", "damage $X$")
    for j in range(18):
        kk = int(np.ceil(tau[j]))
        ks = np.arange(kk + 1)
        xs = np.minimum(X[:kk + 1, j], DD)
        ks = np.r_[ks[:-1], tau[j]]
        L.append(curve("cons!40,line width=0.35pt", xa + ks * sxk, Y1 + xs * sxx))
    med = np.median(X, axis=1)
    km = np.arange(K + 1)[med <= DD]
    L.append(curve("cons,line width=1.0pt", xa + km * sxk, Y1 + med[km] * sxx))
    L.append(r"\draw[intv!85,line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f) node[right,inner sep=1.2pt,text=intv] {$D$};"
             % (xa, Y1 + DD * sxx, xa + wa, Y1 + DD * sxx))
    L.append(r"\node[anchor=south east,align=right,inner sep=0,text=ink!80] at (%.3f,%.3f)"
             r" {steps grow as\\[-1pt]$q(X)^{\gamma}$: convex};" % (xa + wa - 0.08, Y1 + 0.1))
    title(L, xa - 0.6, Y1 + H + 0.18, r"(a) coupled process")
    # (b) the map -----------------------------------------------------------------
    xb, wb = 4.85, 2.75
    sbx, sbz = wb / 3.3, H / 1.12
    frame(L, xb, Y1, wb, H, [(v * sbx, "$%g$" % v) for v in (0, 1, 2, 3)],
          [(v * sbz, "$%g$" % v) for v in (0, 0.5, 1)], r"damage $x$", r"$z$", ylab_dx=0.62)
    #  equal damage steps early and late, projected through the map as L-shaped
    #  bands: the late one arrives on the z axis much thinner
    for x0_, x1_ in ((0.2, 0.7), (2.3, 2.8)):
        xs_ = np.linspace(x0_, x1_, 20)
        za, zb = float(psi(x0_)), float(psi(x1_))
        cv = P(xb + xs_ * sbx, Y1 + psi(xs_) * sbz)
        L.append(r"\fill[alt!14] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f) -- cycle;"
                 % (xb + x0_ * sbx, Y1, cv, xb + x1_ * sbx, Y1))
        L.append(r"\fill[alt!14] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f) -- cycle;"
                 % (xb, Y1 + za * sbz, cv, xb, Y1 + zb * sbz))
        L.append(r"\fill[alt!45] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (xb + x0_ * sbx, Y1, xb + x1_ * sbx, Y1 + 0.06))
        L.append(r"\fill[alt!45] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (xb, Y1 + za * sbz, xb + 0.06, Y1 + zb * sbz))
    xx = np.linspace(0, DD, 120)
    L.append(curve("alt,line width=1.1pt", xb + xx * sbx, Y1 + psi(xx) * sbz))
    L.append(curve("alt!55,densely dashed,line width=0.9pt", xb + xx * sbx, Y1 + psi(xx, 2 * GAM) * sbz))
    for v, lab, col in ((dZ, r"$d_Z$", "alt"), (dV, r"$d_V$", "alt!70")):
        L.append(r"\draw[%s,densely dotted,line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f);"
                 % (col, xb + DD * sbx, Y1, xb + DD * sbx, Y1 + v * sbz, xb, Y1 + v * sbz))
        L.append(r"\node[anchor=west,inner sep=1.2pt,text=%s] at (%.3f,%.3f) {%s};"
                 % (col, xb + DD * sbx + 0.03, Y1 + v * sbz, lab))
    L.append(r"\node[anchor=south east,inner sep=1pt,text=alt!80!black] at (%.3f,%.3f) {$\psi$};"
             % (xb + 1.15 * sbx, Y1 + float(psi(1.15)) * sbz + 0.02))
    L.append(r"\node[anchor=north west,inner sep=1pt,text=alt!65!black] at (%.3f,%.3f) {$\psi_{2\gamma}$};"
             % (xb + 1.5 * sbx, Y1 + float(psi(1.5, 2 * GAM)) * sbz - 0.02))
    title(L, xb - 0.45, Y1 + H + 0.18, r"(b) the map $z=\psi(x)$")
    # (c) Z-space -----------------------------------------------------------------
    xc, wc = 8.55, 3.75
    scz = H / 1.12
    frame(L, xc, Y1, wc, H, [(k * sxk, "$%d$" % k) for k in (0, 20, 40, 60)],
          [(v * scz, "$%g$" % v) for v in (0, 0.5, 1)], "inspection index $k$", r"$Z=\psi(X)$", ylab_dx=0.62)
    lo, hi = np.quantile(Z, 0.05, axis=1), np.quantile(Z, 0.95, axis=1)
    kb = np.arange(K + 1)[hi <= 1.1]
    L.append(r"\fill[alt!13] plot coordinates {%s} -- plot coordinates {%s} -- cycle;"
             % (P(xc + kb * sxk, Y1 + hi[kb] * scz), P(xc + kb[::-1] * sxk, Y1 + lo[kb][::-1] * scz)))
    for j in range(18):
        kk = int(np.ceil(tau[j]))
        ks = np.r_[np.arange(kk), tau[j]]
        zs = np.minimum(Z[:kk + 1, j], dZ)
        L.append(curve("alt!55,line width=0.35pt", xc + ks * sxk, Y1 + zs * scz))
    L.append(r"\draw[intv!85,line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f) node[right,inner sep=1.2pt,text=intv] {$d_Z$};"
             % (xc, Y1 + dZ * scz, xc + wc, Y1 + dZ * scz))
    L.append(r"\draw[acc,densely dashed,line width=0.7pt] (%.3f,%.3f) -- (%.3f,%.3f);"
             % (xc, Y1 + (dZ + kap) * scz, xc + wc, Y1 + (dZ + kap) * scz))
    L.append(r"\node[anchor=south east,inner sep=1.2pt,text=acc!85!black] at (%.3f,%.3f) {$d_Z+\kappa$};"
             % (xc + wc, Y1 + (dZ + kap) * scz))
    L.append(r"\node[anchor=south east,align=right,inner sep=0,text=ink!80] at (%.3f,%.3f)"
             r" {constant drift;\\[-1pt]step variance\\[-1pt]$\propto q(X)^{-\gamma}$};" % (xc + wc - 0.08, Y1 + 0.1))
    title(L, xc - 0.6, Y1 + H + 0.18, r"(c) after the map")
    # (d) first passage -------------------------------------------------------------
    t0, t1 = 38.0, 60.0
    wd, hd = 12.3, 1.55
    sdx = wd / (t1 - t0)
    m2, l2 = (dZ + kap) / mu, (dZ + kap) ** 3 / (s2 * dV)
    m1, l1 = dZ / mu, dZ ** 2 / s2
    ig = lambda m, l: invgauss(mu=m / l, scale=l)
    G2, G1 = ig(m2, l2), ig(m1, l1)
    tt = np.linspace(t0, t1, 260)
    ymax = max(G2.pdf(tt).max(), G1.pdf(tt).max()) * 1.08
    sdy = hd / ymax
    edges = np.arange(t0, t1 + 1e-9, 0.5)
    hist, _ = np.histogram(tau, bins=edges, density=True)
    for e0, hv in zip(edges[:-1], hist):
        if hv > 0:
            L.append(r"\filldraw[fill=alt!22,draw=white,line width=0.3pt] (%.3f,0) rectangle ++(%.3f,%.4f);"
                     % ((e0 - t0) * sdx, 0.5 * sdx, hv * sdy))
    frame(L, 0, 0, wd, hd, [((t - t0) * sdx, "$%d$" % t) for t in range(40, 61, 5)], [],
          r"first-passage index $\tau$ (interpolated)", None, grid=False)
    L.append(curve("alt!85!black,line width=1.0pt", (tt - t0) * sdx, G2.pdf(tt) * sdy))
    L.append(curve("intv,densely dashed,line width=0.9pt", (tt - t0) * sdx, G1.pdf(tt) * sdy))
    ts = np.sort(tau)
    ecdf = lambda g: np.searchsorted(ts, g, side="right") / len(ts)
    grid = np.linspace(t0, t1, 600)
    sup2 = float(np.max(np.abs(ecdf(grid) - G2.cdf(grid))))
    sup1 = float(np.max(np.abs(ecdf(grid) - G1.cdf(grid))))
    lx = (53.3 - t0) * sdx
    L.append(r"\fill[alt!22] (%.3f,%.3f) rectangle ++(0.3,0.13);" % (lx, hd * 0.86))
    L.append(r"\node[anchor=west,inner sep=1.5pt] at (%.3f,%.3f) {%s simulated paths};"
             % (lx + 0.32, hd * 0.86 + 0.065, "{:,}".format(N).replace(",", "\\,")))
    L.append(r"\draw[alt!85!black,line width=1.0pt] (%.3f,%.3f) -- ++(0.3,0) node[right,inner sep=1.5pt,text=ink]"
             r" {second order \eqref{eq:rulcoupled}: $%.3f$};" % (lx, hd * 0.62, sup2))
    L.append(r"\draw[intv,densely dashed,line width=0.9pt] (%.3f,%.3f) -- ++(0.3,0) node[right,inner sep=1.5pt,text=ink]"
             r" {first order, $d_Z$ only: $%.3f$};" % (lx, hd * 0.38, sup1))
    L.append(r"\node[anchor=west,inner sep=1.5pt,text=ink!70] at (%.3f,%.3f) {distance $\sup_\tau|F-F_{\IG}|$};"
             % (lx, hd * 0.14))
    title(L, -0.6, hd + 0.22, r"(d) what the map buys: the first-passage law")
    cap = (r"The Lamperti reduction (\Cref{thm:lamperti}) at $\gamma=1.5$, $\alpha=0.3$,"
           r" $\beta=0.02$, $x_{\rm ref}=1$, $D=3$. (a)~Coupled paths and their median."
           r" (b)~The map $\psi$, and $\psi_{2\gamma}$ for the step variance; shaded,"
           r" equal damage steps early and late. (c)~Paths after the map, with a"
           r" $5$--$95\,\%%$ band. (d)~First-passage index of %s paths against the"
           r" two inverse Gaussian approximations.") % "{:,}".format(N).replace(",", "\\,")
    return tail(L, cap, "fig:lamperti")


# ---------------------------------------------------------------------------
# Figure 4: two distances, and where the second one matters
# ---------------------------------------------------------------------------
def _fd(u, g):
    """Distance to failure in units of x_ref from state 0 to u, exponent g."""
    u = np.asarray(u, float)
    if abs(g) < 1e-12:
        return u
    if abs(g - 1) < 1e-12:
        return np.log1p(u)
    return ((1 + u) ** (1 - g) - 1) / (1 - g)


def infl(u, g):
    return 100 * (np.sqrt(_fd(u, g) / _fd(u, 2 * g)) - 1)


def fig_distances():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Lj = json.load(open(os.path.join(ROOT, "results/theory/lamperti.json"), encoding="utf-8"))["grid"]

    L = []
    head(L)
    H = 2.7
    # (a) integrands -------------------------------------------------------------
    wa = 4.2
    U = 3.0
    sx, sy = wa / 3.25, H / 1.08
    uu = np.linspace(0, U, 160)
    f1, f2 = (1 + uu) ** -GAM, (1 + uu) ** (-2 * GAM)
    L.append(r"\fill[cons!20] (0,0) -- plot coordinates {%s} -- (%.3f,0) -- cycle;"
             % (P(uu * sx, f1 * sy), U * sx))
    L.append(r"\fill[intv!38] (0,0) -- plot coordinates {%s} -- (%.3f,0) -- cycle;"
             % (P(uu * sx, f2 * sy), U * sx))
    frame(L, 0, 0, wa, H, [(0, "$x$"), (1 * sx, "$1$"), (2 * sx, "$2$"), (U * sx, "$D$")],
          [(v * sy, "$%g$" % v) for v in (0, 0.5, 1)], r"state $u$", "integrand")
    L.append(curve("cons,line width=1.1pt", uu * sx, f1 * sy))
    L.append(curve("intv,line width=1.1pt", uu * sx, f2 * sy))
    L.append(r"\draw[ink!60,densely dotted] (%.3f,0) -- (%.3f,%.3f);" % (U * sx, U * sx, H * 0.95))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=cons] at (%.3f,%.3f) {$q(u)^{-\gamma}$};"
             % (1.7 * sx, (2.7 ** -GAM) * sy + 0.03))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=intv] at (%.3f,%.3f) {$q(u)^{-2\gamma}$};"
             % (0.36 * sx, (1.36 ** (-2 * GAM)) * sy + 0.02))
    L.append(r"\node[text=intv!75!black] at (%.3f,%.3f) {$d_V$};" % (0.32 * sx, 0.22 * sy))
    L.append(r"\node[text=cons!80!black,align=center] at (%.3f,%.3f) {$d_Z-d_V$};" % (1.55 * sx, 0.17 * sy))
    dZ, dV = float(_fd(U, GAM)), float(_fd(U, 2 * GAM))
    #  the areas laid out as lengths, on the integrand's own scale
    bx = wa + 0.3
    for i, (v1, col, lab) in enumerate(((dZ, "cons!45", r"$d_Z=%.2f$" % dZ),
                                        (dV, "intv!60", r"$d_V=%.2f$" % dV))):
        L.append(r"\fill[%s] (%.3f,0) rectangle ++(0.22,%.3f);" % (col, bx + i * 0.3, v1 * sy))
        L.append(r"\draw[ink!40,line width=0.3pt] (%.3f,%.3f) -- ++(%.3f,0) node[right,inner sep=1.2pt,text=ink] {%s};"
                 % (bx + i * 0.3 + 0.22, v1 * sy, 0.38 - 0.3 * i, lab))
    title(L, -0.6, H + 0.2, r"(a) two integrals over the same path")
    # (b) theory map --------------------------------------------------------------
    x0, wb = 7.45, 3.85
    u0, u1, g0, g1 = np.log10(0.6), np.log10(20.0), 0.0, 2.4
    UX = lambda u: x0 + (np.log10(u) - u0) / (u1 - u0) * wb
    GY = lambda g: (g - g0) / (g1 - g0) * H
    nx, ny = 44, 34
    ue = np.logspace(u0, u1, nx + 1)
    ge = np.linspace(g0, g1, ny + 1)
    VMAX = 80.0
    for i in range(nx):
        for j in range(ny):
            uc, gc = np.sqrt(ue[i] * ue[i + 1]), 0.5 * (ge[j] + ge[j + 1])
            v = float(infl(uc, gc))
            L.append(r"\fill[cons!%d] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                     % (min(100, int(round(100 * v / VMAX))), UX(ue[i]), GY(ge[j]),
                        UX(ue[i + 1]) + 0.004, GY(ge[j + 1]) + 0.004))
    uf = np.logspace(u0, u1, 200)
    gf = np.linspace(g0 + 1e-6, g1, 160)
    UU, GG = np.meshgrid(uf, gf)
    II = np.vectorize(lambda a, b: float(infl(a, b)))(UU, GG)
    cs = plt.contour(UU, GG, II, levels=[10, 20, 40, 60])
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 5:
                continue
            L.append(curve("white,line width=0.55pt", UX(seg[:, 0]), GY(seg[:, 1])))
            k = int(len(seg) * {10: 0.3, 20: 0.3, 40: 0.3, 60: 0.62}.get(int(lev), 0.3))
            L.append(r"\node[fill=white,fill opacity=0.75,text opacity=1,inner sep=0.6pt,font=\tiny] at (%.3f,%.3f) {$%d\,\%%$};"
                     % (UX(seg[k, 0]), GY(seg[k, 1]), lev))
    plt.close("all")
    frame(L, x0, 0, wb, H, [(UX(u) - x0, "$%g$" % u) for u in (1, 2, 5, 10, 20)],
          [(GY(g), "$%g$" % g) for g in (0, 1, 2)], r"remaining distance $(D-x)/x_{\rm ref}$",
          r"coupling $\gamma$", grid=False, ylab_dx=0.45)
    #  measured points of Table 4: x = 0, D = 3, so u = 3/x_ref
    pts = {}
    for r in Lj:
        if r["gamma"] == 0:
            continue
        pts.setdefault((r["gamma"], 3.0 / r["x_ref"]), []).append(r["sd_overpred_pct_dZ"])
    for (g, u), v in pts.items():
        mv = float(np.mean(v))
        mark(L, UX(u), GY(g), "oo", "intv", 0.06)
        L.append(r"\node[anchor=west,inner sep=1.2pt,font=\tiny,text=black,fill=white,fill opacity=0.8,text opacity=1]"
                 r" at (%.3f,%.3f) {$%.0f$};" % (UX(u) + 0.05, GY(g), mv))
    #  colour bar
    cb = x0 + wb + 0.2
    for k in range(40):
        L.append(r"\fill[cons!%d] (%.3f,%.4f) rectangle ++(0.16,%.4f);" % (int(100 * (k + 0.5) / 40), cb, H * k / 40, H / 40 + 0.003))
    for v in (0, 40, 80):
        L.append(r"\node[anchor=west,inner sep=1pt] at (%.3f,%.3f) {$%d$};" % (cb + 0.17, H * v / VMAX, v))
    L.append(r"\node[rotate=90,anchor=north,inner sep=1pt] at (%.3f,%.3f) {s.d.\ inflation [\%%]};"
             % (cb + 0.75, H / 2))
    title(L, x0 - 0.55, H + 0.2, r"(b) where the second distance matters")
    cap = (r"Two distances to failure. (a)~The integrands of $d_Z$ and $d_V$ at $x=0$,"
           r" $D=3$, $x_{\rm ref}=1$, $\gamma=1.5$; the bars restate the areas as lengths."
           r" (b)~Inflation $\sqrt{d_Z/d_V}-1$ of the RUL standard deviation when the"
           r" spread is carried over $d_Z$; circles are the values measured in"
           r" \Cref{tab:lamperti}.")
    return tail(L, cap, "fig:distances")


# ---------------------------------------------------------------------------
# Figure 5: grid consistency, three parameterisations
# ---------------------------------------------------------------------------
def fig_grid():
    G = json.load(open(os.path.join(ROOT, "results/theory/grid_consistency.json"), encoding="utf-8"))
    T, a0, b0 = G["T"], G["alpha"], G["beta"]
    n = np.array([r["n"] for r in G["naive"]], float)
    dt = T / n
    fuj_m = n * b0 * dt * (1 + (a0 * dt) ** 2 / 2)
    fuj_v = n * (b0 * dt) ** 2 * (a0 * dt) ** 2 * (1 + 1.25 * (a0 * dt) ** 2)
    ser = {
        "cons": ([r["mean"] for r in G["consistent"]], [r["var"] for r in G["consistent"]], "s"),
        "intv": ([r["mean"] for r in G["naive"]], [r["var"] for r in G["naive"]], "o"),
        "ink!60": (list(fuj_m), list(fuj_v), "t"),
    }
    L = []
    head(L)
    w, h = 4.8, 2.15
    pos = {"a": (0.0, 3.6), "b": (6.75, 3.6), "c": (0.0, 0.0), "d": (6.75, 0.0)}
    LX = lambda nn: np.log2(nn) / 6.0 * w

    def logpanel(key, idx, ylo, yhi, ttl, ylab):
        x0, y0 = pos[key]
        Y = lambda v: (np.log10(v) - ylo) / (yhi - ylo) * h
        yt = [(Y(10.0 ** e), "$10^{%d}$" % e) for e in range(int(np.ceil(ylo)), int(np.floor(yhi)) + 1)]
        frame(L, x0, y0, w, h, [(LX(v), "$%d$" % v) for v in (1, 2, 4, 8, 16, 32, 64)], yt,
              r"steps $n$ over the fixed horizon", ylab, ylab_dx=0.78)
        for col, (mm, vv, mk) in ser.items():
            v = np.array([mm, vv][idx], float)
            v = v / v[0]
            yv = Y(v)
            ok = yv >= -1e-9
            xs = x0 + LX(n[ok])
            L.append(curve("%s,line width=0.9pt" % col, xs, y0 + yv[ok]))
            for xx_, yy_ in zip(xs, y0 + yv[ok]):
                mark(L, xx_, yy_, mk, col)
            if not ok.all():
                kk = np.argmax(~ok)
                L.append(r"\draw[%s,line width=0.9pt,densely dotted,%s] (%.3f,%.3f) -- (%.3f,%.3f);"
                         % (col, ARROW, xs[-1], y0 + yv[ok][-1], x0 + LX(n[kk]) - 0.05, y0 + 0.05))
                L.append(r"\node[anchor=south west,inner sep=1pt,text=ink!70] at (%.3f,%.3f) {$\times%s$ at $n=%d$};"
                         % (x0 + LX(n[kk]) - 0.02, y0 + 0.05, sci(1 / v[-1]), n[-1]))
        title(L, x0 - 0.62, y0 + h + 0.18, ttl)
        return Y

    def sci(v):
        e = int(np.floor(np.log10(v)))
        return r"%.1f\cdot10^{%d}" % (v / 10 ** e, e) if e >= 3 else "%.1f" % v

    Ya = logpanel("a", 0, -1.8, 0.4, r"(a) mean: what point accuracy sees",
                  r"$\E[X(T)]$, relative")
    L.append(r"\node[anchor=south west,inner sep=1pt,text=ink!70] at (%.3f,%.3f) {$\div%.1f$};"
             % (pos["a"][0] + LX(64) - 0.6, pos["a"][1] + Ya(fuj_m[-1] / fuj_m[0]) + 0.05, fuj_m[0] / fuj_m[-1]))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=ink!80] at (%.3f,%.3f) {consistent and interval scaling coincide};"
             % (pos["a"][0] + 0.1, pos["a"][1] + Ya(1.0) + 0.06))
    Yb = logpanel("b", 1, -2.4, 0.4, r"(b) variance: what an interval sees",
                  r"$\Var[X(T)]$, relative")
    L.append(r"\node[anchor=north east,inner sep=1pt,text=intv!85!black] at (%.3f,%.3f) {$1/n$};"
             % (pos["b"][0] + LX(16), pos["b"][1] + Yb(1 / 16.0) - 0.04))
    # (c) predictive spread ------------------------------------------------------
    fp = G["first_passage"]
    sp = np.array([r["steps_per_unit"] for r in fp["naive"]], float)
    LC = lambda v: np.log2(v) / 4.0 * w
    x0, y0 = pos["c"]
    YC = lambda v: v / 1.15 * h
    frame(L, x0, y0, w, h, [(LC(v), "$%d$" % v) for v in (1, 2, 4, 8, 16)],
          [(YC(v), "$%g$" % v) for v in (0, 0.5, 1)], "inspections per unit time",
          r"relative", ylab_dx=0.5)
    for key, col, mk in (("naive", "intv", "o"), ("consistent", "cons", "s")):
        sd = np.array([r["sd"] for r in fp[key]]); me = np.array([r["mean"] for r in fp[key]])
        L.append(curve("%s,line width=0.9pt" % col, x0 + LC(sp), y0 + YC(sd / sd[0])))
        L.append(curve("%s!60,densely dashed,line width=0.7pt" % col, x0 + LC(sp), y0 + YC(me / me[0])))
        for a_, b_ in zip(x0 + LC(sp), y0 + YC(sd / sd[0])):
            mark(L, a_, b_, mk, col)
    sdn = np.array([r["sd"] for r in fp["naive"]])
    L.append(r"\draw[acc,line width=0.6pt,{Stealth[length=3pt]}-{Stealth[length=3pt]}] (%.3f,%.3f) -- (%.3f,%.3f);"
             % (x0 + LC(16) - 0.12, y0 + YC(sdn[-1] / sdn[0]), x0 + LC(16) - 0.12, y0 + YC(1.0)))
    L.append(r"\node[anchor=east,inner sep=1.5pt,text=acc!85!black] at (%.3f,%.3f) {$\times%.1f$ narrower};"
             % (x0 + LC(16) - 0.14, y0 + YC(0.55), sdn[0] / sdn[-1]))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=ink!75] at (%.3f,%.3f) {solid: s.d.\ of $\tau$; dashed: mean};"
             % (x0 + 0.08, y0 + 0.05))
    title(L, x0 - 0.62, y0 + h + 0.18, r"(c) the first passage to $D$")
    # (d) magnified consistent spread -----------------------------------------------
    res = G["rul_sd_residual"]
    x0, y0 = pos["d"]
    sdc = np.array([r["sd"] for r in fp["consistent"]])
    lo, hi = 1.318, 1.372
    YD = lambda v: (v - lo) / (hi - lo) * h
    frame(L, x0, y0, w, h, [(LC(v), "$%d$" % v) for v in (1, 2, 4, 8, 16)],
          [(YD(v), "$%.2f$" % v) for v in (1.33, 1.35, 1.37)], "inspections per unit time",
          r"s.d.\ of $\tau$", ylab_dx=0.75)
    lg = np.log2(sp)
    A = np.vstack([np.ones_like(lg), lg]).T
    c0, c1 = np.linalg.lstsq(A, sdc, rcond=None)[0]
    L.append(r"\draw[cons!45,line width=0.8pt] (%.3f,%.3f) -- (%.3f,%.3f);"
             % (x0 + LC(1), y0 + YD(c0), x0 + LC(16), y0 + YD(c0 + 4 * c1)))
    e = 1.96 * res["mc_se_of_each_sd"]
    for a_, v in zip(sp, sdc):
        L.append(r"\draw[cons,line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f);"
                 % (x0 + LC(a_), y0 + YD(v - e), x0 + LC(a_), y0 + YD(v + e)))
        for vv in (v - e, v + e):
            L.append(r"\draw[cons,line width=0.6pt] (%.3f,%.3f) -- ++(0.1,0);" % (x0 + LC(a_) - 0.05, y0 + YD(vv)))
        mark(L, x0 + LC(a_), y0 + YD(v), "s", "cons")
    L.append(r"\node[anchor=south east,align=right,inner sep=1pt,text=ink!80] at (%.3f,%.3f)"
             r" {$+%.1f\,\%%$ over the range; slope $t=%.1f$\\[-1pt]bars: $\pm1.96$ Monte Carlo s.e.};"
             % (x0 + w, y0 + 0.06, 100 * (sdc[-1] / sdc[0] - 1), res["t"]))
    title(L, x0 - 0.62, y0 + h + 0.18, r"(d) the consistent spread, magnified")
    # legend --------------------------------------------------------------------------
    ly = -0.95
    items = (("cons", "s", r"moment-consistent, \Cref{thm:consistent}"),
             ("intv", "o", r"interval scaling \eqref{eq:naive}"),
             ("ink!60", "t", r"both parameters indexed \citep{Fujita2021}"))
    for xx, (col, mk, lab) in zip((-0.2, 4.6, 8.1), items):
        L.append(r"\draw[%s,line width=0.9pt] (%.3f,%.3f) -- ++(0.45,0);" % (col, xx, ly))
        mark(L, xx + 0.225, ly, mk, col)
        L.append(r"\node[anchor=west,inner sep=1.5pt] at (%.3f,%.3f) {%s};" % (xx + 0.5, ly, lab))
    cap = (r"Three time parameterisations of BS on the grid study of \Cref{tab:grid},"
           r" each quantity relative to its coarsest-grid value. (a)~Accumulated mean."
           r" (b)~Accumulated variance. (c)~Simulated first passage to $D=12$: spread"
           r" solid, mean dashed. (d)~The consistent spread magnified, with"
           r" $\pm1.96$ Monte Carlo standard errors.")
    return tail(L, cap, "fig:grid")


# ---------------------------------------------------------------------------
# Figure 6: the real data
# ---------------------------------------------------------------------------
def fig_realdata():
    import pyreadr
    R = json.load(open(os.path.join(ROOT, "results/realdata/realdata.json"), encoding="utf-8"))
    AC = json.load(open(os.path.join(ROOT, "results/realdata/alloy_clock.json"), encoding="utf-8"))["sheppard"]
    DB = json.load(open(os.path.join(ROOT, "results/realdata/deviceb_clock.json"), encoding="utf-8"))["all"]
    rd = lambda f: next(iter(pyreadr.read_r(os.path.join(ROOT, "data_external/smrd", f)).values()))
    las, alo = rd("gaaslaser.RData"), rd("alloya.RData")
    L = []
    head(L)
    w, h = 3.38, 2.0
    X0 = (0.0, 4.55, 9.1)
    Y = 0.0
    # (a) lasers --------------------------------------------------------------------
    x0 = X0[0]
    sx, sy = w / 4000.0, h / 14.0
    frame(L, x0, Y, w, h, [(t * sx, "$%d$" % t) for t in (0, 2000, 4000)],
          [(v * sy, "$%d$" % v) for v in (0, 5, 10)], "hours", r"increase [\%]", ylab_dx=0.48)
    for _, g in las.groupby("unit"):
        g = g.sort_values("hours")
        L.append(curve("ink!55,line width=0.4pt", x0 + g.hours.values * sx, Y + np.minimum(g.increase.values, 14) * sy))
    L.append(r"\draw[intv!80,densely dashed,line width=0.6pt] (%.3f,%.3f) -- ++(%.3f,0);" % (x0, Y + 10 * sy, w))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=intv] at (%.3f,%.3f) {failure};" % (x0 + 0.05, Y + 10 * sy))
    title(L, x0 - 0.5, Y + h + 0.18, r"(a) GaAs lasers")
    # (b) Alloy-A ------------------------------------------------------------------
    x0 = X0[1]
    ylo = 0.85
    sx, sy = w / 0.12, h / (float(alo.inches.max()) * 1.02 - ylo)
    frame(L, x0, Y, w, h, [(t * sx, "$%g$" % t) for t in (0, 0.04, 0.08, 0.12)],
          [((v - ylo) * sy, "$%g$" % v) for v in (0.9, 1.2, 1.5)], r"million cycles", "crack length [in]", ylab_dx=0.5)
    for _, g in alo.groupby("specimen"):
        g = g.sort_values("megacycles")
        L.append(curve("acc!75,line width=0.4pt", x0 + g.megacycles.values * sx, Y + (g.inches.values - ylo) * sy))
    title(L, x0 - 0.5, Y + h + 0.18, r"(b) Alloy-A cracks")
    # (c) growth exponents, all estimates --------------------------------------------
    x0 = X0[2]
    BX = lambda b: x0 + (b + 0.2) / 2.4 * w
    frame(L, x0, Y, w, h, [(BX(b) - x0, "$%g$" % b) for b in (0, 0.5, 1, 1.5, 2)], [],
          r"growth exponent $b$", None, grid=False)
    for b, col, lab in ((1.0, "cons", "moment-\\\\[-1pt]consistent"), (2.0, "intv", "interval\\\\[-1pt]scaling")):
        L.append(r"\draw[%s,line width=0.7pt,densely dashed] (%.3f,%.3f) -- (%.3f,%.3f);" % (col, BX(b), Y, BX(b), Y + h * 0.80))
        L.append(r"\node[anchor=south,align=center,inner sep=1pt,text=%s] at (%.3f,%.3f) {%s};" % (col, BX(b), Y + h * 0.80, lab))
    rows = ((R["R1_laser"], "ink!70", "o", "lasers"),
            (DB, "alt", "t", "Device-B"),
            (AC, "acc", "s", "Alloy-A, clock"),
            (R["R1_alloy_detrended"], "acc!60", "s", "Alloy-A, trend"))
    for k, (d, col, mk, lab) in enumerate(rows):
        yy = Y + h * (0.66 - 0.17 * k)
        L.append(r"\draw[%s,line width=1.1pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (col, BX(d["b_lo"]), yy, BX(d["b_hi"]), yy))
        mark(L, BX(d["b"]), yy, mk, col, 0.07)
        L.append(r"\node[anchor=west,inner sep=1pt,font=\tiny] at (%.3f,%.3f) {%s};" % (BX(d["b_hi"]) + 0.06, yy, lab))
    title(L, x0 - 0.5, Y + h + 0.18, r"(c) growth exponent")
    cap = (r"Real degradation data \citep{MeekerEscobar1998}. (a),\,(b)~Two of the records."
           r" (c)~Exponent $b$ of the within-unit variance growth with $95\,\%$ bootstrap"
           r" intervals; consistency predicts $b=1$, interval scaling $b=2$.")
    return tail(L, cap, "fig:realdata")


# ---------------------------------------------------------------------------
# Figure 7: end-to-end application to the lasers
# ---------------------------------------------------------------------------
def _app_fan(L, x0, Y, w, h, t, x, tk, bands, X, Yv, D, xt, yt, xlab, ylab, ttl, legend_xy):
    """One unit: readings (filled to the origin, hollow after) and 90 % bands."""
    sx = lambda v: x0 + (v - X[0]) / (X[1] - X[0]) * w
    sy = lambda v: Y + (np.clip(v, Yv[0], Yv[1]) - Yv[0]) / (Yv[1] - Yv[0]) * h
    frame(L, x0, Y, w, h, [(sx(v) - x0, lab) for v, lab in xt],
          [(sy(v) - Y, lab) for v, lab in yt], xlab, ylab, ylab_dx=0.48)
    k = int(np.argmin(np.abs(t - tk)))
    tf = t[k:]
    for (lo, hi), col, sty in zip(bands, ("cons", "intv"),
                                  ("line width=0.5pt", "line width=0.6pt,densely dashed")):
        L.append(r"\fill[%s,opacity=%.2f] plot coordinates {%s} -- plot coordinates {%s} -- cycle;"
                 % (col, 0.16 if col == "cons" else 0.10, P(sx(tf), sy(lo)), P(sx(tf[::-1]), sy(hi[::-1]))))
        L.append(curve("%s,%s" % (col, sty), sx(tf), sy(lo)))
        L.append(curve("%s,%s" % (col, sty), sx(tf), sy(hi)))
    L.append(r"\draw[ink!50,densely dashed,line width=0.5pt] (%.3f,%.3f) -- ++(%.3f,0);" % (x0, sy(D), w))
    L.append(r"\node[anchor=south west,inner sep=1pt,text=ink!70] at (%.3f,%.3f) {failure};" % (x0 + 0.05, sy(D)))
    L.append(r"\draw[ink!35,line width=0.4pt] (%.3f,%.3f) -- ++(0,%.3f);" % (sx(tk), Y, h))
    L.append(r"\node[anchor=north west,inner sep=1pt,text=ink!70] at (%.3f,%.3f) {now};" % (sx(tk), Y + h))
    for ti, xi in zip(t, x):
        mark(L, sx(ti), sy(xi), "o" if ti <= tk + 1e-9 else "oo", "ink", 0.042)
    lx, ly = legend_xy
    mark(L, x0 + lx + 0.14, Y + ly, "oo", "ink", 0.042)
    L.append(r"\node[anchor=west,inner sep=1pt,text=ink!70] at (%.3f,%.3f) {observed};" % (x0 + lx + 0.25, Y + ly))
    for j, (col, sty, lab) in enumerate((("cons", "line width=0.8pt", "consistent"),
                                         ("intv", "line width=0.8pt,densely dashed", "interval"))):
        L.append(r"\draw[%s,%s] (%.3f,%.3f) -- ++(0.28,0) node[right,inner sep=1.5pt,text=%s] {%s};"
                 % (col, sty, x0 + lx, Y + ly - 0.24 * (j + 1), col, lab))
    title(L, x0 - 0.5, Y + h + 0.18, ttl)


def _app_cover(L, x0, Y, w, h, R, key, lab, ttl, legend):
    """Coverage of the 90 % interval, both directions, BS and Weibull."""
    c0, c1 = 50.0, 100.0
    CY = lambda c: Y + (c - c0) / (c1 - c0) * h
    frame(L, x0, Y, w, h, [], [(CY(c) - Y, "$%d$" % c) for c in (50, 60, 70, 80, 90, 100)],
          None, r"coverage [\%]", ylab_dx=0.48)
    L.append(r"\draw[ink!50,densely dashed,line width=0.5pt] (%.3f,%.3f) -- ++(%.3f,0);" % (x0, CY(90), w))
    L.append(r"\node[anchor=north,inner sep=1.5pt,text=ink!70] at (%.3f,%.3f) {nominal};" % (x0 + w / 2, CY(90)))
    dirs = []
    for r in R:
        d = (r["h_ref"], r["h_dep"])
        if d not in dirs:
            dirs.append(d)
    gx = {dirs[0]: 0.22 * w, dirs[1]: 0.78 * w}
    for r in R:
        g = x0 + gx[(r["h_ref"], r["h_dep"])]
        dx = -0.12 if r["law"] == "BS" else 0.12
        for m, col in (("consistent", "cons"), ("interval", "intv")):
            v = 100 * r[m]["cover90"]
            mark(L, g + dx + (-0.2 if m == "consistent" else 0.2), CY(min(v, 99.5)),
                 "o" if r["law"] == "BS" else "s", col, 0.065)
    for (a_, b_), g in gx.items():
        L.append(r"\node[anchor=north,inner sep=1.5pt] at (%.3f,%.3f) {$%s\to%s$};"
                 % (x0 + g, Y, key(a_), key(b_)))
    L.append(r"\node[anchor=north,inner sep=1pt] at (%.3f,%.3f) {records $\to$ monitoring [%s]};"
             % (x0 + w / 2, Y - 0.40, lab))
    if legend:
        mark(L, x0 + w / 2 - 0.55, CY(76), "o", "ink!60", 0.055)
        L.append(r"\node[anchor=west,inner sep=1pt] at (%.3f,%.3f) {BS};" % (x0 + w / 2 - 0.48, CY(76)))
        mark(L, x0 + w / 2 - 0.55, CY(68), "s", "ink!60", 0.055)
        L.append(r"\node[anchor=west,inner sep=1pt] at (%.3f,%.3f) {Weibull};" % (x0 + w / 2 - 0.48, CY(68)))
    title(L, x0 - 0.5, Y + h + 0.18, ttl)


def _app_bars(L, x0, Y, w, h, R, key, lab, ttl):
    """Warning given by the replacement rule, in inspection intervals (BS)."""
    shades = {-1: "intv", 1: "ink!22", 2: "ink!45", 3: "ink!70"}
    rows = []
    for r in R[:2]:
        for m in ("consistent", "interval"):
            Q = r[m]["policy"]
            c = [min(round((q["T_obs"] - q["t_rep"]) / r["h_dep"]), 3) if np.isfinite(q["t_rep"]) else -1
                 for q in Q]
            rows.append((m, {k_: c.count(k_) for k_ in shades}, len(Q)))
    bh, lx = 0.24, 0.95
    ys = [Y + h - 0.42, Y + h - 0.72, Y + h - 1.40, Y + h - 1.70]
    for (m, cnt, n), y in zip(rows, ys):
        L.append(r"\node[anchor=east,inner sep=1pt,text=%s] at (%.3f,%.3f) {%s};"
                 % ("cons" if m == "consistent" else "intv", x0 + lx - 0.04, y, m))
        xx_ = x0 + lx
        for k_, col in shades.items():
            ww = cnt[k_] / n * (w - lx)
            if ww <= 0:
                continue
            L.append(r"\fill[%s] (%.3f,%.3f) rectangle ++(%.3f,%.3f);" % (col, xx_, y - bh / 2, ww, bh))
            if cnt[k_] >= 3:
                L.append(r"\node[inner sep=0,text=%s,font=\tiny] at (%.3f,%.3f) {%d};"
                         % ("ink" if k_ == 1 else "white", xx_ + ww / 2, y, cnt[k_]))
            xx_ += ww
        L.append(r"\draw[white,line width=0.6pt] (%.3f,%.3f) rectangle ++(%.3f,%.3f);" % (x0 + lx, y - bh / 2, w - lx, bh))
    for y, r in ((ys[0], R[0]), (ys[2], R[1])):
        L.append(r"\node[anchor=south west,inner sep=1pt,text=ink!80] at (%.3f,%.3f) {$%s\to%s$ %s};"
                 % (x0 + lx, y + bh / 2 + 0.02, key(r["h_ref"]), key(r["h_dep"]), lab))
    lg = x0 + 0.15
    for k_, lb in ((-1, "missed"), (1, "$1$"), (2, "$2$"), (3, r"$\ge3$")):
        L.append(r"\fill[%s] (%.3f,%.3f) rectangle ++(0.16,0.16);" % (shades[k_], lg, Y - 0.30))
        L.append(r"\node[anchor=west,inner sep=1pt] at (%.3f,%.3f) {%s};" % (lg + 0.18, Y - 0.22, lb))
        lg += 1.25 if k_ == -1 else 0.60
    L.append(r"\node[anchor=north,inner sep=1pt] at (%.3f,%.3f) {inspection intervals of warning};"
             % (x0 + w / 2, Y - 0.40))
    title(L, x0 - 0.5, Y + h + 0.18, ttl)


def fig_application():
    import study_application as SA
    import study_application_alloy as SAA
    import study_realdata as SR
    RL = json.load(open(os.path.join(ROOT, "results/realdata/application.json"), encoding="utf-8"))
    RA = json.load(open(os.path.join(ROOT, "results/realdata/application_alloy.json"), encoding="utf-8"))
    L = []
    head(L)
    w, h = 3.38, 2.0
    X0 = (0.0, 4.55, 9.1)
    YT, YB = 3.55, 0.0
    # lasers: unit 10, records every 1000 h, monitored every 250 h, origin 2000 h
    Pl = SR.paths(SR.load("gaaslaser"), "unit", "hours", "increase")
    unit, tk, hr, hd = 9, 2000.0, 1000.0, 250.0
    t, x = Pl[unit]
    cv2r = SA.cv2_pooled([p for j, p in enumerate(Pl) if j != unit], hr)
    k = int(np.nonzero(t == tk)[0][0])
    bands = []
    for cv2 in (cv2r * hr / hd, cv2r):
        SA.RNG = np.random.default_rng(5)
        S = SA.simulate("BS", x[k], tk, hd, cv2, len(t) - 1 - k)
        bands.append((np.r_[x[k], np.quantile(S, 0.05, axis=0)], np.r_[x[k], np.quantile(S, 0.95, axis=0)]))
    _app_fan(L, X0[0], YT, w, h, t, x, tk, bands, (0, 4000), (0, 14), 10.0,
             [(v, "$%d$" % v) for v in (0, 2000, 4000)], [(v, "$%d$" % v) for v in (0, 5, 10)],
             "hours", r"increase [\%]", r"(a) laser, $90\,\%$ bands", (1.75, 0.66))
    _app_cover(L, X0[0], YB, w, h, RL, lambda v: "%d" % v, "h", r"(d) laser coverage", True)
    # Device-B: device 109 (195 C), records every 500 h, monitored every 125 h, origin 1000 h
    import study_application_deviceb as SDB
    RD = json.load(open(os.path.join(ROOT, "results/realdata/application_deviceb.json"), encoding="utf-8"))
    Pd = SDB.load()
    unit, tk, hr = 8, 1000.0, 500.0
    temp, t, x = Pd[unit]
    train = [q for j, q in enumerate(Pd) if j != unit and q[0] == temp]
    pc_, pu = SDB.clock_p(train), SDB.unit_p(train)
    F = SDB.fit(train, hr, pc_)
    k = int(np.nonzero(t == tk)[0][0])
    bands = []
    for m in ("consistent", "interval"):
        SDB.RNG = np.random.default_rng(5)
        SDB.SA.RNG = SDB.RNG
        S = SDB.simulate("BS", m, t[:k + 1], x[:k + 1], t[k + 1:], pc_, F[m], F["df"], pu)
        bands.append((np.r_[x[k], np.quantile(S, 0.05, axis=0)], np.r_[x[k], np.quantile(S, 0.95, axis=0)]))
    _app_fan(L, X0[1], YT, w, h, t, x, tk, bands, (0, 2000), (0, 1.6), 0.5,
             [(v, "$%d$" % v) for v in (0, 1000, 2000)], [(v, "$%.1f$" % v) for v in (0, 0.5, 1.0, 1.5)],
             "hours", "power drop [dB]", r"(b) Device-B, $90\,\%$ bands", (0.12, 1.85))
    _app_cover(L, X0[1], YB, w, h, RD, lambda v: "%d" % v, "h", r"(e) Device-B coverage", False)
    # Alloy-A: specimen 2, records every 20 kcycles, monitored every 10, origin 40
    Pa = SR.paths(SR.load("alloya"), "specimen", "megacycles", "inches")
    unit, tk, hr, hd = 1, 0.04, 0.02, 0.01
    t, x = Pa[unit]
    F = SAA.fit([p for j, p in enumerate(Pa) if j != unit], hr)
    k = int(np.argmin(np.abs(t - tk)))
    bands = []
    for m in ("consistent", "interval"):
        SAA.RNG = np.random.default_rng(5)
        S = SAA.simulate("BS", m, x[k], t[:k + 1], x[:k + 1], hd, hr, F["gamma"], F[m], len(t) - 1 - k,
                         F["df"], F[m + "_sig2"])
        bands.append((np.r_[x[k], np.quantile(S, 0.05, axis=0)], np.r_[x[k], np.quantile(S, 0.95, axis=0)]))
    _app_fan(L, X0[2], YT, w, h, 1000 * t, x, 1000 * tk, bands, (0, 120), (0.85, 1.8), 1.6,
             [(v, "$%d$" % v) for v in (0, 40, 80, 120)], [(v, "$%.1f$" % v) for v in (1.0, 1.2, 1.4, 1.6)],
             "kcycles", "crack length [in]", r"(c) Alloy-A, $90\,\%$ bands", (1.95, 0.66))
    _app_cover(L, X0[2], YB, w, h, RA, lambda v: "%d" % round(1000 * v), "kcycles", r"(f) Alloy-A coverage", False)
    cap = (r"End-to-end application: lasers, Device-B and Alloy-A cracks. Top, one unit"
           r" with $90\,\%$ bands, coarse records and fine monitoring. Bottom, coverage of the"
           r" $90\,\%$ level interval, BS and Weibull steps.")
    return tail(L, cap, "fig:application")
