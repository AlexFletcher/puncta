#!/usr/bin/env python3
"""
puncta_modelling_manuscript.py
==============================

Single-file generator for every results figure in

    "Clustering versus sorting: a mass-conserving reaction-diffusion model
     of planar polarity puncta"

All figures are written to ``figures/`` (override with --outdir).

Figure  ->  function                      ->  output file
   2,3      fig2_3_dispersion_nullcline()      AB_dispersion.png, AB_nullcline.png
   4,5      fig4_5_full_six_species()          AB_numeric_turing.png, AB_numeric_pinning.png
   6        fig6_pinned_front()                AB_pinning_maxwell.png
   7        fig7_spike()                       AB_spike.png
   8        fig8_competition()                 AB_competition.png
   9        fig9_coarsening()                  AB_coarsening.png
   10       fig10_two_reservoir()              AB_tworeservoir.png
   11       fig11_polarity()                   AB_polarity.png

Usage
-----
    python3 puncta_modelling_manuscript.py            # all figures
    python3 puncta_modelling_manuscript.py 2 3 7      # only the named figures
    python3 puncta_modelling_manuscript.py --outdir /tmp/figs 7

"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import brentq
from scipy.linalg import eig
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
#  Global style + output directory                                            #
# --------------------------------------------------------------------------- #
# Set USETEX=True to typeset every label through a real LaTeX installation
# (exact Computer Modern + the manuscript's math); this needs `latex` + `dvipng`
# (and ghostscript) on PATH. The default (False) uses Matplotlib's built-in
# mathtext with the Computer Modern font set: no external tools, and visually
# almost identical for these figures. Override at the command line with --usetex.
USETEX = False


def configure_style(usetex=USETEX):
    """Apply the figure style; LaTeX-match the manuscript via usetex or mathtext-CM."""
    rc = {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "font.family": "serif",
        "axes.unicode_minus": False,   # cmr10 has no Unicode minus glyph
        "axes.formatter.use_mathtext": True,
    }
    if usetex:
        rc.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })
    else:
        rc.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",                       # Computer Modern math
            "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
        })
    plt.rcParams.update(rc)


configure_style()
OUTDIR = "figures"


def _save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")
    plt.close(fig)


# =========================================================================== #
#  CONFIG  -- every tunable parameter lives here, grouped by regime           #
# =========================================================================== #
# Feedback strengths (one per regime).
ALPHA_ONSET = 0.14    # c^2, onset/dispersion (Figs 2, 4)
ALPHA_SPIKE = 0.30    # c^2, spike / array / polarity (Figs 7-11)
ALPHA_SAT = 0.60      # saturating Hill, mesa regime (Figs 3-right, 5, 6)
SAT_M, SAT_RHO = 2, 0.1

# Geometry.
ELL_ONSET = 100.0     # Figs 2, 4, 5  (Da~100 << ell^2 -> many coexisting interfaces)
ELL_SPIKE = 40.0      # Figs 6, 7, 10b, 11  (single isolated structure)
N_ONSET = 3.9         # a_T = a^dag_T = b_T = b^dag_T for the onset family

# Complex diffusivity is the non-dimensionalisation reference (D_c = 1) everywhere.


# =========================================================================== #
#  Shared numerics                                                            #
# =========================================================================== #
def neumann_laplacian(NX, dx, dense=False):
    """Second-order 1-D Laplacian with zero-flux (Neumann) ends."""
    main = -2.0 * np.ones(NX)
    main[0] = main[-1] = -1.0
    off = np.ones(NX - 1)
    L = sp.diags([off, main, off], [-1, 0, 1], format="csc") / dx**2
    return L.toarray() if dense else L


def imex_factorizations(L, dt, diffusivities):
    """Return splu factorizations of (I - dt*D*L) for each D in *diffusivities*."""
    NX = L.shape[0]
    I = sp.identity(NX, format="csc")
    return [splu((I - dt * D * L).tocsc()) for D in diffusivities]


def feedback(kind):
    """Return (K, Kp) for 'turing' (c^2) or 'sat' (Hill c^m/(1+rho c^m))."""
    if kind == "turing":
        return (lambda c: np.clip(c, 0, None)**2,
                lambda c: 2 * np.clip(c, 0, None))
    m, rho = SAT_M, SAT_RHO
    K = lambda c: np.clip(c, 0, None)**m / (1 + rho * np.clip(c, 0, None)**m)
    Kp = lambda c: (m * np.clip(c, 0, None)**(m - 1)) / (1 + rho * np.clip(c, 0, None)**m)**2
    return K, Kp


def wellmixed_suss(alpha, n, kind, V=1.0):
    """Highest physical root c* in (0,n) of alpha*K(c)*(n-c)^2 = V*c."""
    K, _ = feedback(kind)
    cs = np.linspace(1e-6, n - 1e-6, 4000)
    F = alpha * K(cs) * (n - cs)**2 - V * cs
    roots = [brentq(lambda c: alpha * K(np.array(c)) * (n - c)**2 - V * c, cs[i], cs[i + 1])
             for i in range(len(cs) - 1) if F[i] * F[i + 1] < 0]
    return max(roots) if roots else 0.5 * n


def spike_reservoir(alpha, ell, n, K=1):
    """Tall-branch monomer reservoir abar for K mass-limited spikes (c^2 feedback)."""
    a_fold = (K * 12.0 / (alpha * ell))**(1.0 / 3)
    return brentq(lambda a: a * ell + K * 6.0 / (alpha * a**2) - n * ell, 1e-4, a_fold)


# =========================================================================== #
#  Figures 2 & 3 : dispersion relation + reactive nullclines (analytical)     #
# =========================================================================== #
def fig2_3_dispersion_nullcline():
    """Linear theory of the symmetric decoupled triplet (V'=0, a_T=b^dag_T)."""
    n = N_ONSET

    def block_entries(cstar, alpha, K, Kp, V=1.0):
        a = b = n - cstar
        return alpha * b * K(cstar), alpha * a * K(cstar), V - alpha * a * b * Kp(cstar)

    def cubic_max_re(q2, k1, k2, phi, Da, Db):
        c2 = (k1 + k2 + phi) + (Da + Db + 1.0) * q2
        c1 = ((Da * Db + Da + Db) * q2**2
              + (Da * (k2 + phi) + Db * (k1 + phi) + (k1 + k2)) * q2)
        c0 = Da * Db * q2**2 * (q2 + k1 / Da + k2 / Db + phi)
        return np.roots([1.0, c2, c1, c0]).real.max()

    crit_q2 = lambda k1, k2, phi, Da, Db: -(phi + k1 / Da + k2 / Db)

    # ---- Figure 2: dispersion (Turing feedback) ----
    K_T, Kp_T = feedback("turing")
    cstar = wellmixed_suss(ALPHA_ONSET, n, "turing")
    k1, k2, phi = block_entries(cstar, ALPHA_ONSET, K_T, Kp_T)
    print("=== Dispersion (Turing K=c^2, V=1) ===")
    print(f"alpha={ALPHA_ONSET}, n={n}, c*={cstar:.4f}, kappa1=kappa2={k1:.4f}, phi={phi:.4f}")

    ell = ELL_ONSET
    qq = np.linspace(0, 1.2, 1200)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {"100": "#1f77b4", "10": "#d62728"}
    qc_store = {}
    for Db, label, col in [(100.0, r"$D_b = D_a = 100$", colors["100"]),
                           (10.0, r"$D_b = 10,\ D_a = 100$", colors["10"])]:
        Da = 100.0
        sig = np.array([cubic_max_re(q**2, k1, k2, phi, Da, Db) for q in qq])
        qc = np.sqrt(crit_q2(k1, k2, phi, Da, Db))
        qc_store[col] = qc
        ax.plot(qq, sig, color=col, lw=2, label=label)
        ax.axvline(qc, color=col, ls=":", lw=1.2)
        print(f"  Da={Da:.0f}, Db={Db:.0f}: q_c={qc:.4f}, "
              f"unstable modes={int(np.floor(qc * ell / np.pi))}")
    nmax = int(np.floor(1.2 * ell / np.pi))
    qn = np.array([p * np.pi / ell for p in range(1, nmax + 1)])
    ax.plot(qn, np.zeros_like(qn), marker="|", ls="none", color="0.45",
            markersize=8, label=r"modes $q_n=n\pi/\ell,\ \ell=100$")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel=r"wavenumber $q$", ylabel=r"$\max\ \mathrm{Re}\,\sigma(q)$", xlim=(0, 1.2))
    ax.set_title(r"Dispersion relation, feedback $\mathcal{K}(c)=c^2$")
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    ymax = ax.get_ylim()[1]
    ax.text(qc_store[colors["100"]], ymax * 0.97, r"$q_c$", color=colors["100"],
            ha="center", va="top", fontsize=12)
    ax.text(qc_store[colors["10"]], ymax * 0.83, r"$q_c$", color=colors["10"],
            ha="center", va="top", fontsize=12)
    fig.tight_layout()
    _save(fig, "AB_dispersion.png")

    # ---- Figure 3: reactive nullclines (Turing vs saturating) ----
    def nullcline(c, alpha, K, V=1.0):
        return c + np.sqrt(V * c / (alpha * K(c)))

    def Fprime(c, nn, alpha, K, Kp, V=1.0):
        return alpha * Kp(c) * (nn - c)**2 - 2 * alpha * K(c) * (nn - c) - V

    K_WP, Kp_WP = feedback("sat")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    panels = [(axes[0], r"Turing feedback $\mathcal{K}(c)=c^{2}$", ALPHA_ONSET, K_T, Kp_T),
              (axes[1], r"Wave-pinning feedback $\mathcal{K}(c)=c^{m}/(1+\rho c^{m})$",
               ALPHA_SAT, K_WP, Kp_WP)]
    for ax, title, alpha, K, Kp in panels:
        cc = np.linspace(1e-4, n - 1e-4, 4000)
        nn = nullcline(cc, alpha, K)
        stable = Fprime(cc, nn, alpha, K, Kp) < 0
        ax.plot(np.where(stable, nn, np.nan), cc, color="#2a7", lw=2.4, label="stable")
        ax.plot(np.where(~stable, nn, np.nan), cc, color="#c44", lw=2.0, ls="--", label="unstable")
        ax.axhline(0, color="#2a7", lw=2.4)              # c=0 is a stable equilibrium
        ax.axvline(n, color="0.5", ls=":", lw=1.2)
        ax.text(n + 0.04, n * 0.93, r"$n=3.9$", color="0.4", fontsize=12)
        ax.set(xlabel=r"conserved total density $n\ (=a_T=b^{\dagger}_T)$", xlim=(0, 5.2),
               ylim=(-0.15, n))
        ax.set_title(title, fontsize=12)
        dN = np.gradient(nn, cc)
        folds = [(round(nn[i], 3), round(cc[i], 3))
                 for i in np.where(np.diff(np.sign(dN)) != 0)[0]]
        print(f"\n=== Nullcline: {title} (alpha={alpha}) ===\nfolds (n,c): {folds}")
    axes[0].set_ylabel(r"steady complex concentration $c^{*}$")
    axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    _save(fig, "AB_nullcline.png")


# =========================================================================== #
#  Figures 4 & 5 : full six-species PDE from a near-uniform state             #
# =========================================================================== #
def fig4_5_full_six_species():
    ELL, NX, dt, T = ELL_ONSET, 400, 0.002, 100.0
    n = N_ONSET
    x = (np.arange(NX) + 0.5) * ELL / NX
    Lap = neumann_laplacian(NX, ELL / NX)

    def smooth_perturbation(seed, amp):
        rng = np.random.default_rng(seed)
        k = np.arange(1, 41)
        coef = rng.standard_normal(len(k)) / k
        f = sum(coef[j] * np.cos(np.pi * k[j] * x / ELL + 2 * np.pi * rng.random())
                for j in range(len(k)))
        f -= f.mean(); f /= np.abs(f).max()
        return amp * f

    def run(alpha, kind, Da, Db, eps=2.5, seed=1):
        K, _ = feedback(kind)
        luA, luB, luC = imex_factorizations(Lap, dt, [Da, Db, 1.0])
        cstar = wellmixed_suss(alpha, n, kind); astar = n - cstar
        a = np.full(NX, astar); ad = np.full(NX, astar)
        b = np.full(NX, astar); bd = np.full(NX, astar)
        c = cstar + smooth_perturbation(seed, eps)
        cd = cstar + smooth_perturbation(seed + 99, eps)
        m0 = np.array([(a + c).mean(), (ad + cd).mean(), (b + cd).mean(), (bd + c).mean()])
        for _ in range(int(T / dt)):
            R1 = alpha * K(c) * a * bd
            R2 = alpha * K(cd) * ad * b
            a = luA.solve(a + dt * (-R1 + c)); bd = luB.solve(bd + dt * (-R1 + c)); c = luC.solve(c + dt * (R1 - c))
            ad = luA.solve(ad + dt * (-R2 + cd)); b = luB.solve(b + dt * (-R2 + cd)); cd = luC.solve(cd + dt * (R2 - cd))
        m1 = np.array([(a + c).mean(), (ad + cd).mean(), (b + cd).mean(), (bd + c).mean()])
        drift = np.max(np.abs(m1 - m0))
        print(f"  {kind} Da={Da} Db={Db}: max|c|={c.max():.1f}, mass drift={drift:.2e}")
        return dict(a=a, ad=ad, b=b, bd=bd, c=c, cd=cd)

    def panel(ax, S, title):
        ax.plot(x, S["a"], label=r"$a$"); ax.plot(x, S["ad"], label=r"$a^\dagger$")
        ax.plot(x, S["b"], label=r"$b$"); ax.plot(x, S["bd"], label=r"$b^\dagger$")
        ax.plot(x, S["c"], label=r"$c$", lw=2); ax.plot(x, S["cd"], label=r"$c^\dagger$", lw=2)
        ax.set(xlabel="$x$", ylabel="concentration", xlim=(0, ELL)); ax.set_title(title)

    def make(kind, alpha, fname, flabel):
        print(f"== {kind} (alpha={alpha}) ==")
        Ssym = run(alpha, kind, 100, 100, seed=1)
        Sasy = run(alpha, kind, 100, 10, seed=1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
        panel(a1, Ssym, r"(a) symmetric $D_a=D_b=100$")
        panel(a2, Sasy, r"(b) asymmetric $D_a=100,\ D_b=10$")
        a1.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        fig.suptitle(fr"{flabel},  $\alpha={alpha}$,  $a_T=\dots=b^\dagger_T={n}$,  "
                     fr"$\ell={ELL:.0f}$,  $t={T:.0f}$", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, f"{fname}.png")

    make("turing", ALPHA_ONSET, "AB_numeric_turing", r"Turing feedback $\mathcal{K}(c)=c^2$")
    make("sat", ALPHA_SAT, "AB_numeric_pinning", r"Wave-pinning feedback $\mathcal{K}(c)=c^2/(1+0.1c^2)$")


# =========================================================================== #
#  Figure 6 : pinned front (Maxwell sharp-interface theory vs simulation)     #
# =========================================================================== #
def fig6_pinned_front():
    alpha = ALPHA_SAT
    K, _ = feedback("sat")
    s = np.sqrt(SAT_RHO)
    K_int = lambda C: (C - np.arctan(s * C) / s) / SAT_RHO
    cplus = brentq(lambda C: K_int(C) - 0.5 * C * K(C), 1e-6, 50.0)
    Pstar = cplus / (alpha * K(cplus)); abar = np.sqrt(Pstar)
    print("=== Maxwell pinned-front prediction (Hill feedback) ===")
    print(f"alpha={alpha}, m={SAT_M}, rho={SAT_RHO}: c+*={cplus:.4f}, sqrt(P*)={abar:.4f}, "
          f"window {abar:.3f}<n<{abar + cplus:.3f}")

    N_TOTAL = N_ONSET                         
    xf_pred = (N_TOTAL - abar) / cplus      
    print(f"chosen n={N_TOTAL:.4f} -> predicted xf/ell={xf_pred:.4f}")

    ELL, NX, dt, TMAX = ELL_SPIKE, 400, 0.01, 3000.0
    Da = Db = 1.0e5                            # sqrt(Da)>>ell -> monomers well mixed (load-bearing)
    dx = ELL / NX
    x = (np.arange(NX) + 0.5) * dx
    Lap = neumann_laplacian(NX, dx)
    luC, luA = imex_factorizations(Lap, dt, [1.0, Da])

    c = np.clip(0.02 + 0.5 * cplus * (1 - np.tanh((x - 0.45 * ELL) / 1.0)), 0.0, None)
    a = np.full(NX, N_TOTAL - c.mean()); b = a.copy()
    for _ in range(int(TMAX / dt)):
        Rc = alpha * K(c) * a * b - c
        c = luC.solve(c + dt * Rc); a = luA.solve(a - dt * Rc); b = luA.solve(b - dt * Rc)

    plateau = c[x < 0.12 * ELL].mean(); monomer = a.mean()
    half = 0.5 * (c.max() + c.min()); above = c > half
    xf_sim = x[above][-1] if above.any() else 0.0
    print(f"sim: <a+c>={(a + c).mean():.4f}, plateau={plateau:.4f} (c+*={cplus:.4f}), "
          f"monomer={monomer:.4f} (sqrtP*={abar:.4f}), xf/ell={xf_sim / ELL:.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, c, color="#6a3d9a", lw=2.2, label=r"$c(x)$ (simulation)")
    ax.plot(x, a, color="#1f77b4", lw=1.8, label=r"$a(x)=b^{\dagger}(x)$ (simulation)")
    ax.axhline(cplus, color="#6a3d9a", ls="--", lw=1.3, label=r"Maxwell plateau $c_{+}^{*}$")
    ax.axhline(abar, color="#1f77b4", ls="--", lw=1.3, label=r"Maxwell monomer $\sqrt{P^{*}}$")
    ax.axvline(xf_pred * ELL, color="0.4", ls=":", lw=1.4, label=r"predicted front $x_f$")
    ax.set(xlabel="$x$", ylabel="concentration", xlim=(0, ELL))
    ax.set_title(r"Pinned front: sharp-interface prediction vs. simulation")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.tight_layout()
    _save(fig, "AB_pinning_maxwell.png")


# =========================================================================== #
#  Figure 7 : mass-limited spike (single punctum), theory vs simulation       #
# =========================================================================== #
def fig7_spike():
    alpha, ELL, n = ALPHA_SPIKE, ELL_SPIKE, 2.5      # K=1 spike at abar=0.5 => cmax=20
    a_fold = (12.0 / (alpha * ELL))**(1.0 / 3)
    n_fold = (a_fold * ELL + 6.0 / (alpha * a_fold**2)) / ELL
    abar_tall = spike_reservoir(alpha, ELL, n, K=1)
    print(f"single-spike fold n_fold={n_fold:.4f}; here n={n}, abar_tall={abar_tall:.4f}, "
          f"cmax={3 / (2 * alpha * abar_tall**2):.3f}")

    NX, dx, dt, D, TMAX = 400, ELL / 400, 0.004, 2000.0, 400.0
    x = (np.arange(NX) + 0.5) * dx
    Lap = neumann_laplacian(NX, dx)
    luC, luA = imex_factorizations(Lap, dt, [1.0, D])

    c = 8.0 * np.exp(-((x - ELL / 2) / 1.0)**2)
    a = np.full(NX, n - c.mean()); b = a.copy()
    for _ in range(int(TMAX / dt)):
        r = alpha * c**2 * a * b - c
        c = luC.solve(c + dt * r); a = luA.solve(a - dt * r); b = luA.solve(b - dt * r)

    cp_sim, abar_sim = c.max(), a.mean()
    cp_pred = 3.0 / (2 * alpha * abar_sim**2)
    idx = np.where(c > cp_sim / 2)[0]; fwhm = (idx[-1] - idx[0]) * dx
    print(f"sim: abar={abar_sim:.4f}, cmax={cp_sim:.4f} (pred {cp_pred:.4f}), "
          f"FWHM={fwhm:.3f} (pred {4 * np.arccosh(np.sqrt(2)):.3f})")

    xc = x[np.argmax(c)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, c, color="#6a3d9a", lw=2.2, label=r"$c(x)$ (simulation)")
    ax.plot(x, cp_pred / np.cosh((x - xc) / 2)**2, color="k", ls="--", lw=1.3,
            label=r"$\frac{3}{2\alpha\bar a^2}\,\mathrm{sech}^2(\frac{x-x_0}{2})$")
    ax.plot(x, a, color="#1f77b4", lw=1.8, label=r"$a(x)=b^{\dagger}(x)$ (simulation)")
    ax.axhline(abar_sim, color="#1f77b4", ls=":", lw=1.2, label=r"$\bar a$ (measured)")
    ax.set(xlabel="$x$", ylabel="concentration", xlim=(0, ELL))
    ax.set_title(r"Mass-limited spike (puncta), $\mathcal{K}(c)=c^2$: theory vs simulation")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    _save(fig, "AB_spike.png")


# =========================================================================== #
#  Figure 8 : competition of puncta (coarsening + rate vs Da)                 #
# =========================================================================== #
def fig8_competition():
    alpha = ALPHA_SPIKE
    ELL, N = ELL_SPIKE, 4.5   
    dt = 0.01
    cp = 3.0 / (2 * alpha * spike_reservoir(alpha, ELL, N, K=2)**2)  
    Rc = lambda a, b, c: alpha * c**2 * a * b - c

    def build(NX, dx, D):
        return imex_factorizations(neumann_laplacian(NX, dx), dt, [1.0, D])

    # (a) two-spike coarsening kymograph -- coarsening is ~3x slower at ell=40, so the
    #     window/snapshots are retuned (extinction near t~280 instead of t~90).
    NX = 200; dx = ELL / NX; x = (np.arange(NX) + 0.5) * dx; D = 50.0
    luC, luA = build(NX, dx, D)
    c = cp * 1.10 / np.cosh((x - ELL / 4) / 2)**2 + cp * 0.90 / np.cosh((x - 3 * ELL / 4) / 2)**2
    a = np.full(NX, N - c.mean()); b = a.copy()
    targets = [0, 60, 150, 280, 450, 680]; TMAX = 720.0  
    tset = {int(round(t / dt)) for t in targets}
    snaps, tsnap = [], []
    for step in range(int(TMAX / dt)):
        if step in tset:
            snaps.append(c.copy()); tsnap.append(step * dt)
        r = Rc(a, b, c); c = luC.solve(c + dt * r); a = luA.solve(a - dt * r); b = luA.solve(b - dt * r)
    snaps = np.array(snaps); tsnap = np.array(tsnap)

    # (b) competition rate vs Da via first-e-fold (robust across the slower aligned rates).
    def rate(D, delta=0.01, TMAX=500.0, NX=160):
        dx = ELL / NX; x = (np.arange(NX) + 0.5) * dx
        luC, luA = build(NX, dx, D)
        c = cp * (1 + delta) / np.cosh((x - ELL / 4) / 2)**2 + cp * (1 - delta) / np.cosh((x - 3 * ELL / 4) / 2)**2
        a = np.full(NX, N - c.mean()); b = a.copy()
        d0 = abs(c[x < ELL / 2].sum() - c[x >= ELL / 2].sum()) * dx
        for step in range(int(TMAX / dt)):
            r = Rc(a, b, c); c = luC.solve(c + dt * r); a = luA.solve(a - dt * r); b = luA.solve(b - dt * r)
            if abs(c[x < ELL / 2].sum() - c[x >= ELL / 2].sum()) * dx > np.e * d0:
                return 1.0 / ((step + 1) * dt)
        return np.nan

    Ds = np.array([50., 100., 200., 400.])     
    rates = np.array([rate(D) for D in Ds])   
    print("competition rates:", dict(zip(Ds, np.round(rates, 6))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for sshot, t, col in zip(snaps, tsnap, plt.cm.viridis(np.linspace(0, 0.9, len(snaps)))):
        ax1.plot(x, sshot, lw=1.8, color=col, label=f"$t={t:.0f}$")
    ax1.set(xlabel="$x$", ylabel="$c(x,t)$", xlim=(0, ELL))
    ax1.set_title(r"(a) Two-spike coarsening ($D_a=50$)")
    ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    ax2.semilogy(Ds, rates, "o-", color="#6a3d9a", lw=1.8, ms=7)
    ax2.set(xlabel=r"monomer diffusivity $D_a$", ylabel=r"competition rate $\lambda_{\mathrm{comp}}$")
    ax2.set_title(r"(b) Metastability: $\lambda_{\mathrm{comp}}$ collapses with $D_a$")
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "AB_competition.png")


# =========================================================================== #
#  Figure 9 : metastable coarsening of a multi-spike array                    #
# =========================================================================== #
def fig9_coarsening():
    alpha, ELL, N0 = ALPHA_SPIKE, 60.0, N_ONSET      # ell=60 needed to host K0=6 spikes
    NX, dt, K0 = 480, 0.008, 6
    dx = ELL / NX; x = (np.arange(NX) + 0.5) * dx
    Lap = neumann_laplacian(NX, dx)
    Rc = lambda a, b, c: alpha * c**2 * a * b - c

    def seed(rng):
        abar = spike_reservoir(alpha, ELL, N0, K=K0)
        cp = 3.0 / (2 * alpha * abar**2)
        c = np.zeros(NX)
        for xc in ELL * (np.arange(K0) + 0.5) / K0:
            c += cp * (1 + 0.04 * rng.standard_normal()) / np.cosh((x - xc) / 2)**2
        a = np.full(NX, N0 - c.mean()); b = a.copy()
        return a, b, c

    count = lambda c: len(find_peaks(c, height=1.0, distance=12)[0])

    def evolve(D, TMAX, rng, save_kymo=False):
        luC, luA = imex_factorizations(Lap, dt, [1.0, D])
        a, b, c = seed(rng)
        ts, Ns, kymo, ktimes = [], [], [], []
        for step in range(int(TMAX / dt)):
            r = Rc(a, b, c); c = luC.solve(c + dt * r); a = luA.solve(a - dt * r); b = luA.solve(b - dt * r)
            if step % 250 == 0:
                ts.append(step * dt); Ns.append(count(c))
                if save_kymo and step % 1000 == 0:
                    kymo.append(c.copy()); ktimes.append(step * dt)
            if step % 5000 == 0 and count(c) == 1 and step * dt > 50:
                ts.append(step * dt); Ns.append(1); break
        return np.array(ts), np.array(Ns), (np.array(kymo), np.array(ktimes))

    _, _, (kymo, ktimes) = evolve(80.0, 6000.0, np.random.default_rng(3), save_kymo=True)
    curves = {}
    for D in (40.0, 80.0, 160.0):
        t, Nt, _ = evolve(D, 6000.0, np.random.default_rng(3))
        curves[D] = (t, Nt)
        print(f"D={D:5.0f}: N {Nt[0]}->{Nt[-1]} by t={t[-1]:.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    T, Xg = np.meshgrid(ktimes, x, indexing="ij")
    pc = ax1.pcolormesh(Xg, T, kymo, shading="auto", cmap="magma")
    ax1.set(xlabel="$x$", ylabel="time $t$")
    ax1.set_title(r"(a) Metastable coarsening of puncta ($D_a=80$)")
    fig.colorbar(pc, ax=ax1, label="$c(x,t)$")
    for D, (t, Nt) in curves.items():
        ax2.step(t, Nt, where="post", lw=2, label=f"$D_a={D:.0f}$")
    ax2.set(xlabel="time $t$", ylabel="number of puncta $N(t)$")
    ax2.set_title("(b) Smaller $D_a$ retains more puncta (metastable)")
    ax2.set_yticks(range(0, K0 + 1)); ax2.legend(frameon=False); ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "AB_coarsening.png")


# =========================================================================== #
#  Figure 10 : two-reservoir effects (harmonic-mean law + single-spike spectrum)
# =========================================================================== #
def fig10_two_reservoir():
    # ---- (a) competition rate vs harmonic mean (asymmetric diffusion) ----
    alpha_a, ELL_a, N_a = ALPHA_SPIKE, ELL_SPIKE, 4.5  
    NXa, dta = 240, 0.002                             
    dxa = ELL_a / NXa; xa = (np.arange(NXa) + 0.5) * dxa
    LapA = neumann_laplacian(NXa, dxa)
    cp_a = 3.0 / (2 * alpha_a * spike_reservoir(alpha_a, ELL_a, N_a, K=2)**2)   # = 20 (was the 0.8 guess)

    def efold(Da, Db, delta=0.01, TMAX=300.0):
        luC, luA, luB = imex_factorizations(LapA, dta, [1.0, Da, Db])
        c = cp_a * (1 + delta) / np.cosh((xa - ELL_a / 4) / 2)**2 + cp_a * (1 - delta) / np.cosh((xa - 3 * ELL_a / 4) / 2)**2
        a = np.full(NXa, N_a - c.mean()); b = a.copy()
        d0 = abs(c[xa < ELL_a / 2].sum() - c[xa >= ELL_a / 2].sum()) * dxa
        for step in range(int(TMAX / dta)):
            r = alpha_a * c**2 * a * b - c
            c = luC.solve(c + dta * r); a = luA.solve(a - dta * r); b = luB.solve(b - dta * r)
            if abs(c[xa < ELL_a / 2].sum() - c[xa >= ELL_a / 2].sum()) * dxa > np.e * d0:
                return 1.0 / ((step + 1) * dta)
        return np.nan

    pairs = [(400, 400), (800, 400), (400, 800), (1600, 400), (400, 1600), (800, 800), (1600, 800), (1600, 1600)]
    hm = np.array([Da * Db / (Da + Db) for Da, Db in pairs])
    lam = np.array([efold(float(Da), float(Db)) for Da, Db in pairs])
    abar = spike_reservoir(alpha_a, ELL_a, N_a, K=2)
    slope = 2 * alpha_a * abar**3 / (3 * ELL_a)
    print(f"abar(2-spike)={abar:.4f}, parameter-free slope={slope:.4e}")

    # ---- (b) single-spike spectrum swept over Da/Db ----
    alpha, ELL, n = ALPHA_SPIKE, ELL_SPIKE, 2.5
    NX, dt = 200, 0.005
    dx = ELL / NX; x = (np.arange(NX) + 0.5) * dx
    LapS = neumann_laplacian(NX, dx)
    Lap_dense = LapS.toarray()

    def steady(Da, Db, T=500.0):
        luC, luA, luB = imex_factorizations(LapS, dt, [1.0, Da, Db])
        c = 3.0 / (2 * alpha * 0.5**2) / np.cosh((x - ELL / 2) / 2)**2
        a = np.full(NX, n - c.mean()); b = a.copy()
        for _ in range(int(T / dt)):
            r = alpha * c**2 * a * b - c
            c = luC.solve(c + dt * r); a = luA.solve(a - dt * r); b = luB.solve(b - dt * r)
        return a, b, c

    def spec(Da, Db):
        a, b, c = steady(Da, Db)
        c2 = c**2
        Raa, Rab, Rac = -alpha * c2 * b, -alpha * c2 * a, -2 * alpha * c * a * b + 1.0
        Rca, Rcb, Rcc = alpha * c2 * b, alpha * c2 * a, 2 * alpha * c * a * b - 1.0
        J = np.zeros((3 * NX, 3 * NX))
        J[:NX, :NX] = Da * Lap_dense + np.diag(Raa); J[:NX, NX:2 * NX] = np.diag(Rab); J[:NX, 2 * NX:] = np.diag(Rac)
        J[NX:2 * NX, :NX] = np.diag(Raa); J[NX:2 * NX, NX:2 * NX] = Db * Lap_dense + np.diag(Rab); J[NX:2 * NX, 2 * NX:] = np.diag(Rac)
        J[2 * NX:, :NX] = np.diag(Rca); J[2 * NX:, NX:2 * NX] = np.diag(Rcb); J[2 * NX:, 2 * NX:] = Lap_dense + np.diag(Rcc)
        w = eig(J, right=False); w = w[np.argsort(-w.real)]; nz = w[np.abs(w) > 1e-3]
        lr = next((l for l in nz if abs(l.imag) < 1e-4), np.nan)
        cpx = [l for l in nz if l.imag > 1e-4]
        return c.max(), lr, (cpx[0] if cpx else complex(np.nan, np.nan))

    Da = 100.0; Dbs = [100, 25, 6, 3, 2, 1.5]
    reals, cpxs, ratios = [], [], []
    for Db in Dbs:
        _, lr, lc = spec(Da, float(Db)); ratios.append(Da / Db); reals.append(lr); cpxs.append(lc)
    ratios = np.array(ratios)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot([0, hm.max() * 1.05], [0, slope * hm.max() * 1.05], "k--", lw=1.3,
             label=r"$\lambda_{\rm comp}=\frac{2\alpha\bar a^3}{3\ell}\frac{D_aD_b}{D_a+D_b}$ ($\gamma=\ell/4$)")
    ax1.scatter(hm, lam, c="#6a3d9a", s=55, zorder=3)
    for (Da_, Db_), l in zip(pairs, lam):
        ax1.annotate(f"({Da_},{Db_})", (Da_ * Db_ / (Da_ + Db_), l), fontsize=7.5,
                     xytext=(4, -3), textcoords="offset points")
    ax1.set(xlabel=r"$D_aD_b/(D_a+D_b)$  (harmonic mean)",
            ylabel=r"competition rate $\lambda_{\rm comp}$", xlim=(0, None), ylim=(0, None))
    ax1.set_title(r"(a) Parameter-free prediction ($\gamma=\ell/4$)")
    ax1.legend(frameon=False, fontsize=10, loc="upper left")

    sc = ax2.scatter([l.real for l in cpxs], [l.imag for l in cpxs], c=np.log10(ratios),
                     cmap="viridis", s=70, zorder=3, label="oscillatory pair")
    ax2.scatter([l.real for l in cpxs], [-l.imag for l in cpxs], c=np.log10(ratios), cmap="viridis", s=70, zorder=3)
    ax2.scatter([l.real for l in reals], [0] * len(reals), c=np.log10(ratios), cmap="viridis",
                marker="s", s=55, edgecolors="k", linewidths=0.4, zorder=4, label="amplitude mode (real)")
    ax2.axvline(0, color="#d62728", lw=1.4)
    ax2.text(0.01, 0.12, r"Re$\,\lambda=0$", color="#d62728", fontsize=9)
    fig.colorbar(sc, ax=ax2).set_label(r"$\log_{10}(D_a/D_b)$")
    ax2.set(xlabel=r"$\mathrm{Re}\,\lambda$", ylabel=r"$\mathrm{Im}\,\lambda$", xlim=(None, 0.18))
    ax2.set_title(r"(b) No Hopf as $D_a/D_b$ grows")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left"); ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "AB_tworeservoir.png")


# =========================================================================== #
#  Figure 11 : spike-level polarity (drift law + segregation/merging)         #
# =========================================================================== #
def fig11_polarity():
    alpha, ELL, n, DM = ALPHA_SPIKE, ELL_SPIKE, 2.5, 1000.0
    NX, dt = 300, 0.003
    dx = ELL / NX; x = (np.arange(NX) + 0.5) * dx
    Lap = neumann_laplacian(NX, dx)
    luC, luM = imex_factorizations(Lap, dt, [1.0, DM])

    def relax(xc, T=40.0):
        c = 20.0 / np.cosh((x - xc) / 2)**2; a = np.full(NX, n - c.mean())
        for _ in range(int(T / dt)):
            r = alpha * c**2 * a**2 - c; c = luC.solve(c + dt * r); a = luM.solve(a - dt * r)
        return a, c

    shift = lambda f, d: np.interp(x - d, x, f, left=f[0], right=f[-1])
    cen = lambda c: (x * np.clip(c, 0, None)).sum() / np.clip(c, 0, None).sum()
    a0, c0 = relax(ELL / 2); print(f"c_max={c0.max():.2f}")

    def step(a, ad, c, cd, kappa):
        Vc, Vcd = 1 + kappa * cd, 1 + kappa * c
        c = luC.solve(c + dt * (alpha * c**2 * a**2 - Vc * c)); a = luM.solve(a + dt * (-alpha * c**2 * a**2 + Vc * c))
        cd = luC.solve(cd + dt * (alpha * cd**2 * ad**2 - Vcd * cd)); ad = luM.solve(ad + dt * (-alpha * cd**2 * ad**2 + Vcd * cd))
        return a, ad, c, cd

    # (b) drift law
    pred, meas = [], []
    for kappa in (-0.012, 0.012):
        for d in (3.0, 3.5, 4.0):
            a, ad, c, cd = a0.copy(), a0.copy(), shift(c0, -d / 2), shift(c0, +d / 2)
            xc0 = cen(c); j = int(round(xc0 / dx)); cdp = (cd[j + 1] - cd[j - 1]) / (2 * dx)
            ts, xs = [0.0], [xc0]
            for sidx in range(int(1.5 / dt)):
                a, ad, c, cd = step(a, ad, c, cd, kappa)
                if sidx % 25 == 0:
                    ts.append((sidx + 1) * dt); xs.append(cen(c))
            pred.append(-2.5 * kappa * cdp); meas.append(np.polyfit(ts, xs, 1)[0])
    pred, meas = np.array(pred), np.array(meas)

    # (a) trajectories
    def traj(kappa, xi0, T):
        a, ad, c, cd = a0.copy(), a0.copy(), shift(c0, +xi0), shift(c0, -xi0)
        ts, xc, xd = [0.0], [cen(c)], [cen(cd)]
        for sidx in range(int(T / dt)):
            a, ad, c, cd = step(a, ad, c, cd, kappa)
            if sidx % 40 == 0:
                ts.append((sidx + 1) * dt); xc.append(cen(c)); xd.append(cen(cd))
        return np.array(ts), np.array(xc), np.array(xd)

    tP, xcP, xdP = traj(+0.012, 0.2, 26.0)
    tN, xcN, xdN = traj(-0.012, 3.0, 26.0)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
    axA.plot(tP, xcP, color="#6a3d9a", lw=2, label=r"$c$"); axA.plot(tP, xdP, color="#8c564b", lw=2, label=r"$c^\dagger$")
    axA.plot(tN, xcN, color="#6a3d9a", lw=2, ls="--"); axA.plot(tN, xdN, color="#8c564b", lw=2, ls="--")
    axA.set(xlabel="time $t$", ylabel="punctum position")
    axA.set_title(r"(a) $\mathcal{V}'>0$ segregates (solid); $\mathcal{V}'<0$ merges (dashed)")
    axA.legend(frameon=False, fontsize=11, loc="center right")
    lim = max(np.abs(pred).max(), np.abs(meas).max()) * 1.1
    axB.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="$y=x$")
    axB.scatter(pred, meas, s=55, color="#6a3d9a", zorder=3)
    axB.set(xlabel=r"predicted $-2.5\,\mathcal{V}'\,(c^\dagger)'(x_c)$", ylabel=r"measured drift $\dot x_c$")
    axB.set_title(r"(b) Drift law (varying $\mathcal{V}'$, separation)")
    axB.legend(frameon=False, fontsize=11, loc="upper left"); axB.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "AB_polarity.png")
    print(f"drift-law fit slope={np.polyfit(pred, meas, 1)[0]:.3f} (expect 1.0)")


# =========================================================================== #
#  Driver                                                                     #
# =========================================================================== #
FIGURES = {
    "2": fig2_3_dispersion_nullcline, "3": fig2_3_dispersion_nullcline,
    "4": fig4_5_full_six_species, "5": fig4_5_full_six_species,
    "6": fig6_pinned_front, "7": fig7_spike, "8": fig8_competition,
    "9": fig9_coarsening, "10": fig10_two_reservoir, "11": fig11_polarity,
}


def main():
    parser = argparse.ArgumentParser(description="Generate puncta-model results figures.")
    parser.add_argument("figs", nargs="*", help="figure numbers (default: all)")
    parser.add_argument("--outdir", default="figures", help="output directory")
    parser.add_argument("--usetex", action="store_true",
                        help="render all text via a real LaTeX install (needs dvipng/ghostscript)")
    args = parser.parse_args()

    if args.usetex:
        configure_style(usetex=True)

    global OUTDIR
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    # 2&3 and 4&5 share a function; dedupe while preserving order.
    requested = args.figs or list(FIGURES)
    seen, todo = set(), []
    for f in requested:
        fn = FIGURES.get(f)
        if fn is None:
            print(f"  [skip] unknown figure '{f}'"); continue
        if fn not in seen:
            seen.add(fn); todo.append((f, fn))
    for f, fn in todo:
        print(f"\n--- Figure {f} ({fn.__name__}) ---")
        fn()


if __name__ == "__main__":
    main()