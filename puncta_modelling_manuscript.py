#!/usr/bin/env python3
"""
puncta_modelling_manuscript.py
==============================

Single-file generator for every results figure in

    "Clustering versus sorting: a mass-conserving reaction-diffusion model
     of planar polarity puncta"

All figures are written to ``figures/`` (override with --outdir).

Figure  ->  function                        ->  output file
   2,3      fig2_3_dispersion_nullcline()        AB_dispersion.png, AB_nullcline.png
   4        fig4_full_six_species()              AB_numeric_turing.png,
                                                 AB_numeric_pinning.png
   5        fig5_pinned_front()                  AB_pinning_maxwell.png
   6        fig6_spike()                         AB_spike.png
   7        fig7_competition()                   AB_competition.png
   8        fig8_competition_saturation()        AB_competition_saturation.png   *
   9        fig9_coarsening()                    AB_coarsening.png
   10       fig10_two_reservoir()                AB_tworeservoir.png
   11       fig11_two_reservoir_plane()          AB_tworeservoir_plane.png       *
   12       fig12_sorting()                      AB_sorting.png                  *

   * needs the funpy spectral library (see Dependencies); everything else uses
     NumPy/SciPy/Matplotlib alone.

Figures may be selected by number or by output label, so `8` and
`AB_competition_saturation` are equivalent.

Dependencies
------------
NumPy, SciPy and Matplotlib throughout. Figures 8, 11 and 12 additionally need
funpy, which supplies the ultraspherical resolvent behind the nonlocal
eigenvalue problem and the exponential integrator behind the sorting runs:

    git clone https://github.com/adrs0049/funpy
    cd funpy && pip install .

It is imported only when one of those three figures is requested.

Usage
-----
    python3 puncta_modelling_manuscript.py            # all figures
    python3 puncta_modelling_manuscript.py 2 3 7      # only the named figures
    python3 puncta_modelling_manuscript.py AB_spike   # by label
    python3 puncta_modelling_manuscript.py --outdir /tmp/figs 7

"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
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
ALPHA_SPIKE = 0.30    # c^2, spike / array / sorting (Figs 6-12)
ALPHA_SAT = 0.60      # saturating Hill, mesa regime (Figs 3-right, 4, 5)
SAT_M, SAT_RHO = 2, 0.1

# Geometry.
ELL_ONSET = 100.0     # Figs 2, 4  (Da~100 << ell^2 -> many coexisting interfaces)
ELL_SPIKE = 40.0      # Figs 5-12  (single isolated structure)
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
#  Spectral machinery  --  Figures 8, 11 and 12(a-c) only                     #
# =========================================================================== #
# Figures 8, 11 and 12(a-c) use the funpy spectral library; it is imported only
# when one of them is requested, so the other figures run without it.
NU0 = 1.25          # largest eigenvalue of L0 = d_yy - 1 + 2w


def _require_funpy():
    """Import funpy, or exit with an actionable message."""
    try:
        from funpy import Fun
        from funpy.colloc.chebOp import ChebOp
        from funpy.ivp import PDEOperator, solve_pde
        from funpy.operators.resolvent import Resolvent
    except ImportError as exc:
        raise SystemExit(
            "Figures 8, 11 and 12 need the funpy spectral library, which does "
            "not appear to be installed.\n"
            "    git clone https://github.com/adrs0049/funpy\n"
            "    cd funpy && pip install .\n"
            f"  (import failed: {exc})")
    return Fun, ChebOp, Resolvent, PDEOperator, solve_pde


class SpikeResolvent:
    """P(lambda) = int (L0 - lambda)^{-1} w^2 dy, and the NLEP determinant zeta.

    The puncta couple only through the monomer reservoir, so the eigenvalue
    problem is a rank-one perturbation of L0 = d_yy - 1 + 2w and its determinant
    collapses to zeta(lambda) = 1 + beta mu_m lambda P(lambda) = 0. Only the
    resolvent of L0 is discretised -- by the ultraspherical spectral method on
    the half-line, Neumann at the origin to keep the even subspace carrying w^2.
    Exact anchors P(0) = 6 and P'(0) = 3 hold here to 5e-3.
    """

    def __init__(self, L=30.0, n=400):
        Fun, ChebOp, Resolvent, _, _ = _require_funpy()
        op = ChebOp(domain=[0, L])
        op.eqn = ['diff(u, x, 2) - u + 2*(3/2)*(1/cosh(x/2))**2 * u']
        op.bcs = ["u'(0)", f'u({L})']
        self.L = L
        self.R = Resolvent(op, n=n)
        self.w2 = Fun(op=lambda y: (1.5 / np.cosh(0.5 * y)**2)**2, domain=[0, L])

    def P(self, lam):
        """P(lambda) as a (possibly complex) scalar; w^2 is even, hence the 2."""
        psi = self.R.solve(self.w2, lam, check_residual=False)
        half = complex(np.atleast_1d(np.asarray(np.sum(psi))).ravel()[0])
        return 2.0 * half

    def zeta(self, lam, beta, mu):
        """Rank-one Woodbury determinant of the competition NLEP."""
        return 1.0 + beta * mu * lam * self.P(lam)


# Bracketing grid: dense on (0, 1.2], then refined towards nu_0 = 5/4. It runs
# past the numerical pole of P (at ~1.24979 on the truncated line, not nu_0
# exactly), so zeta has two sign changes and nlep_root takes the first. Correct
# only while the pole stays above every physical root -- re-check if L or n in
# SpikeResolvent change.
_NLEP_GRID = np.concatenate([np.linspace(1e-6, 1.2, 140),
                             NU0 - np.logspace(-1.4, -7, 60)])


def nlep_root(sr, beta, mu):
    """Root of zeta(lambda) = 0, bracketed by scanning towards the pole at nu_0.

    A sign change of zeta is either a zero or the pole of P, and brentq converges
    to both, so the root is checked: a zero leaves a residual ~1e-14, the pole
    1e9-1e13. (P(root) > 0 would not separate them -- at D_a = 1e6 the pole
    bracket lands on the positive side of P.)
    """
    vals = np.array([sr.zeta(g, beta, mu).real for g in _NLEP_GRID])
    idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if not len(idx):
        return np.nan
    i = idx[0]
    root = brentq(lambda L_: sr.zeta(L_, beta, mu).real,
                  _NLEP_GRID[i], _NLEP_GRID[i + 1])
    resid = abs(sr.zeta(root, beta, mu).real)
    if resid > 1e-6:
        raise RuntimeError(
            f"bracket [{_NLEP_GRID[i]:.6f}, {_NLEP_GRID[i + 1]:.6f}] contains "
            f"the pole of P rather than a zero of zeta (residual {resid:.3g}): "
            f"the truncation pole has moved below a physical root -- check L "
            f"and n in SpikeResolvent.")
    return root


def competition_modes(K, ell):
    """Competition eigenvalues, and their mode wavenumbers, for K equal spikes.

    The K-1 eigenpairs of the conserved Neumann Green's matrix H(x_j; x_k) on
    the zero-sum subspace sum_k S_k = 0. The wavenumber m is the number of sign
    changes along the array: m = 1 is one end against the other, m = K-1 is
    strict alternation, i.e. neighbouring puncta trading mass. For K = 2 this
    returns the exact -ell/4 of the two-spike Green's function.
    """
    pos = np.array([(j + 0.5) * ell / K for j in range(K)])
    Xi, Xj = np.meshgrid(pos, pos, indexing="ij")
    H = (np.abs(Xi - Xj) / 2 + (Xi + Xj) / 2
         - (Xi**2 + Xj**2) / (2 * ell) - ell / 3)
    Q = np.linalg.qr(np.eye(K) - np.ones((K, K)) / K)[0][:, :K - 1]
    ev, V = np.linalg.eigh(Q.T @ H @ Q)
    ms = [int(np.sum(np.diff(np.sign(Q @ v)) != 0)) for v in V.T]
    return ev, np.array(ms)


def sorting_pde(alpha, D, L, Vprime, N):
    """Two coupled triplets as an ETDRK4 operator on the periodic domain [0, L].

    Cross-modulated turnover V(c^dag) = 1 + V' c^dag couples the two complex
    orientations through the unbinding rate alone, conserving <a+c> and
    <a^dag+c^dag> separately. Pass L = 2*ell with even initial data from
    `sorting_seed` to obtain the paper's zero-flux problem on [0, ell].
    """
    _, _, _, PDEOperator, _ = _require_funpy()
    from scipy.fft import fft, ifft

    def nonlinear(u_hats, k):
        a = np.real(ifft(u_hats[0])); c = np.real(ifft(u_hats[1]))
        ad = np.real(ifft(u_hats[2])); cd = np.real(ifft(u_hats[3]))
        r1 = alpha * c**2 * a**2
        r2 = alpha * cd**2 * ad**2
        cross = Vprime * c * cd                 # V' c^dag c, symmetric in c<->c^dag
        return [fft(-r1 + c + cross), fft(r1 - cross),
                fft(-r2 + cd + cross), fft(r2 - cross)]

    return PDEOperator(domain=[0, L],
                       linear=lambda k: [-D * k**2, -k**2 - 1.0,
                                         -D * k**2, -k**2 - 1.0],
                       nonlinear=nonlinear, n_vars=4, N=N)


def _half_mean(u, N):
    """Trapezoidal average of an even field over [0, ell] -- the conserved mass.

    A plain mean over the first N/2 + 1 points is not conserved and reports a
    spurious drift of order 1e-2.
    """
    h = N // 2
    return float((0.5 * u[0] + u[1:h].sum() + 0.5 * u[h]) / h)


def sorting_seed(ell, n, N, xc, xd, cmax):
    """Even initial data: a c-punctum at xc, a c^dag-punctum at xd.

    Each carries its mirror image at 2 ell - x, and the reservoirs are set so
    the trapezoidal masses over [0, ell] equal n.
    """
    L = 2.0 * ell
    x = L * np.arange(N) / N

    def bump(x0):
        out = np.zeros(N)
        for xm in (x0, L - x0):
            dx = x - xm
            dx = dx - L * np.round(dx / L)
            out += cmax / np.cosh(0.5 * dx)**2
        return out

    c, cd = bump(xc), bump(xd)
    return [(n - _half_mean(c, N)) * np.ones(N), c,
            (n - _half_mean(cd, N)) * np.ones(N), cd]


def _peak_centre(c, x, ell):
    """Sub-grid punctum position: parabolic fit about the discrete maximum.

    Restricted to the physical half [0, ell] -- the even extension places a
    mirror peak of exactly equal height at 2 ell - x_c, between which an
    unrestricted argmax would hop.
    """
    h = len(c) // 2
    i = int(np.argmax(c[:h + 1]))
    im, ip = max(i - 1, 0), min(i + 1, h)
    y0, y1, y2 = c[im], c[i], c[ip]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    return float(np.clip(x[i] + delta * (x[1] - x[0]), 0.0, ell))


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
#  Figure 4 : full six-species PDE from a near-uniform state                  #
# =========================================================================== #
def fig4_full_six_species():
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
#  Figure 5 : pinned front (Maxwell sharp-interface theory vs simulation)     #
# =========================================================================== #
def fig5_pinned_front():
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
#  Figure 6 : mass-limited spike (single punctum), theory vs simulation       #
# =========================================================================== #
def fig6_spike():
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
#  Figure 7 : competition of puncta (coarsening + rate vs Da)                 #
# =========================================================================== #
def fig7_competition():
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
#  Figure 8 : validity of the closed-form rate, and its K-dependence          #
# =========================================================================== #
# lambda_comp measured from the zero-flux PDE: relax a symmetric two-spike pair,
# kick it antisymmetrically by +-1%, and fit the exponential growth of the
# complex mass difference between the two halves of the junction (one punctum in
# each). Same diagnostic and parameters as Figure 7(b), tabulated rather than
# recomputed because the D_a = 50 run alone takes ~45 min. They approach the NLEP
# root as ell^2/D_a falls -- 0.35 of it at D_a = 50, 1.01 at D_a = 800 -- which is
# the shadow limit switching on.
MEASURED_RATES = {50.0: 0.00540, 100.0: 0.01897, 200.0: 0.05022,
                  400.0: 0.11172, 800.0: 0.21879}


def fig8_competition_saturation():
    alpha, ell, n = ALPHA_SPIKE, ELL_SPIKE, 4.5
    sr = SpikeResolvent(L=30.0, n=400)
    abar2 = spike_reservoir(alpha, ell, n, K=2)
    beta2 = 2.0 / (alpha * abar2**3)
    c_closed, c_nlep, c_meas, c_asym = "#EE6677", "#4477AA", "#228833", "#7B3FA0"

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- (a) where the closed form holds ----
    ax = axes[0]
    Ds = np.logspace(1, 6, 40)
    closed = alpha * abar2**3 * Ds / (3.0 * ell)
    roots = np.array([nlep_root(sr, beta2, -ell / (4.0 * D)) for D in Ds])
    ax.axhline(NU0, color=c_asym, ls=":", lw=1.4)
    ax.text(2e5, NU0 * 1.13, r"$\nu_0=5/4$", color=c_asym, fontsize=8.5, ha="right")
    ax.loglog(Ds, closed, "--", color=c_closed, lw=1.7,
              label=r"closed form $\alpha\bar a^{3}D_a/(3\ell)$")
    ax.loglog(Ds, roots, "-", color=c_nlep, lw=2.2, label="NLEP root")
    md = np.array(sorted(MEASURED_RATES))
    ax.loglog(md, [MEASURED_RATES[d] for d in md], "o", color=c_meas, ms=6,
              label="measured (zero-flux PDE)")
    ax.axvspan(ell**2 / 4.0, Ds.max(), color="0.88", zorder=0)
    ax.text(ell**2 / 4.0 * 1.4, 3e-3, r"$\ell^{2}/D_a\lesssim4$", fontsize=8,
            color="0.35")
    ax.set(xlabel=r"$D_a$", ylabel=r"$\lambda_{\mathrm{comp}}$")
    ax.set_title("(a) the closed form is a two-sided approximation", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    # ---- (b) K-spike arrays ----
    ax = axes[1]
    Ks = list(range(2, 9))
    D_arr = 400.0
    fastest, slowest, sc = [], [], None
    for K in Ks:
        abK = spike_reservoir(alpha, ell, n, K=K)
        betaK = 2.0 / (alpha * abK**3)
        ev, ms = competition_modes(K, ell)
        rates = np.array([nlep_root(sr, betaK, e / D_arr) for e in ev])
        sc = ax.scatter([K] * len(rates), rates, c=ms, cmap=plt.get_cmap("viridis"),
                        vmin=1, vmax=7, s=34, zorder=3, edgecolors="white",
                        linewidths=0.5)
        fastest.append(rates.max()); slowest.append(rates.min())
    ax.plot(Ks, fastest, "-", color="0.35", lw=1.3, zorder=2)
    ax.plot(Ks, slowest, color="0.35", lw=1.3, ls="--", zorder=2)
    ax.axhline(NU0, color=c_asym, ls=":", lw=1.4)
    ax.text(0.04, 0.94, rf"$D_a={D_arr:.0f}$", transform=ax.transAxes,
            fontsize=8.5, color="0.25")
    ax.text(6.15, NU0 * 1.07, r"$\nu_0=5/4$", color=c_asym, fontsize=8.4)
    cb = fig.colorbar(sc, ax=ax, ticks=range(1, 8), pad=0.02)
    cb.set_label("mode wavenumber $m$ (sign changes)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.set(xlabel="$K$ (equally spaced puncta)", ylabel=r"growth rate $\lambda$",
           yscale="log", ylim=(0.015, 2.6), xlim=(1.6, 8.8))
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.tick_params(axis="y", which="minor", left=False)
    ax.set_xticks(Ks)
    ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    ax.set_title(r"(b) all $K-1$ modes unstable; fastest $\to\nu_0$", fontsize=9.5)

    print(f"  abar(2-spike)={abar2:.4f}; NLEP/closed = "
          f"{nlep_root(sr, beta2, -ell / 1600.0) / (alpha * abar2**3 * 400 / (3 * ell)):.3f} at D_a=400")
    fig.tight_layout()
    _save(fig, "AB_competition_saturation.png")


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
#  Figure 11 : the (D_a, D_b) plane -- the slower monomer sets the rate       #
# =========================================================================== #
# Each monomer is transported by its own Green's function while a punctum
# consumes both together, so they enter through the inverse sum
# mu_m = -gamma (1/(abar D_a) + 1/(bbar D_b)) -- hence a harmonic mean rather
# than an additive combination. gamma = ell/4 is the exact Neumann self/cross
# difference for two spikes at ell/4 and 3ell/4.
def fig11_two_reservoir_plane():
    alpha, ell, n = ALPHA_SPIKE, ELL_SPIKE, 4.5
    gamma = ell / 4.0
    sr = SpikeResolvent(L=30.0, n=400)
    abar = bbar = spike_reservoir(alpha, ell, n, K=2)

    # P(lambda) depends on neither the parameters nor mu_m, so G = lambda P is
    # universal: tabulate it once and invert by interpolation rather than
    # re-rooting the NLEP at each of the 36 100 plane points. G increases
    # monotonically from 0 to the pole, so the inversion is a single interp.
    lam_max = 1.2494
    lam = np.unique(np.concatenate([np.linspace(1e-6, 1.15, 260),
                                    lam_max - np.logspace(np.log10(lam_max - 1.15),
                                                          -5, 140)]))
    G = np.array([float(np.real(l * sr.P(l))) for l in lam])
    keep = [0]                       # locate the numerical pole, do not assume it
    for i in range(1, len(G)):
        if not np.isfinite(G[i]) or G[i] <= G[keep[-1]]:
            break
        keep.append(i)
    lam_g, G_g = lam[keep], G[keep]

    def lam_exact(Da, Db):
        mu = -gamma * (1.0 / (abar * Da) + 1.0 / (bbar * Db))
        target = np.atleast_1d(np.asarray(-alpha * abar * bbar / mu, dtype=float))
        out = np.interp(target, G_g, lam_g, left=0.0, right=np.nan)
        out[target > G_g[-1]] = NU0                  # beyond the tabulated pole
        return out.reshape(np.shape(Da))

    def lam_closed(Da, Db):
        return (alpha * (abar * bbar)**2 / (6 * gamma)
                * Da * Db / (abar * Da + bbar * Db))

    fig, axes = plt.subplots(1, 2, figsize=(9.9, 3.9))

    # ---- (a) the plane ----
    ax = axes[0]
    g = np.logspace(1.3, 4.5, 190)
    DA, DB = np.meshgrid(g, g, indexing="ij")
    LAM = lam_exact(DA, DB)
    levels = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.15])
    pcm = ax.pcolormesh(DA, DB, LAM, shading="auto", cmap="viridis",
                        vmin=0, vmax=NU0)
    # viridis runs dark-blue -> yellow, so plain white lines vanish at the top
    # end: stroke them, and use red rather than blue for the closed form.
    stroke = [pe.withStroke(linewidth=1.9, foreground="0.15", alpha=0.6)]
    cs = ax.contour(DA, DB, LAM, levels=levels, colors="white", linewidths=1.0)
    cs.set_path_effects(stroke)
    for lbl in ax.clabel(cs, fmt="%.2g", fontsize=6.5, inline=True, colors="white"):
        lbl.set_path_effects(stroke)
    ax.contour(DA, DB, lam_closed(DA, DB), levels=levels, colors="#EE6677",
               linewidths=1.0, linestyles=":")
    ax.plot(g, g, color="white", lw=0.9, ls="--", alpha=0.75, path_effects=stroke)
    ax.set(xscale="log", yscale="log", xlabel="$D_a$", ylabel="$D_b$")
    ax.set_title("(a) level sets are the harmonic-mean hyperbolae", fontsize=9.5)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\lambda_{\mathrm{comp}}$", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    # ---- (b) the ceiling ----
    # D_b = 100, 400, 1600 spans the range over which the linearised ceiling
    # 2 alpha abar^3 D_b/(3 ell) goes from accurate (0.061 vs 0.063) to badly
    # wrong (0.586 vs 1.000), as the ceiling itself climbs towards nu_0 = 5/4.
    ax = axes[1]
    Da_line = np.logspace(1.3, 5.5, 260)
    for Db_v, col in zip((100.0, 400.0, 1600.0), ("#4477AA", "#228833", "#CCBB44")):
        ax.loglog(Da_line, lam_exact(Da_line, Db_v), color=col, lw=2.1,
                  label=rf"$D_b={Db_v:.0f}$")
        ax.loglog(Da_line, lam_closed(Da_line, Db_v), ":", color=col, lw=1.2)
        ax.axhline(2 * alpha * abar**3 * Db_v / (3 * ell), color=col, lw=0.7,
                   ls="--", alpha=0.5)
    ax.axhline(NU0, color="#7B3FA0", ls=":", lw=1.4)
    ax.text(1.6e5, NU0 * 1.1, r"$\nu_0=5/4$", color="#7B3FA0", fontsize=8,
            ha="right")
    ax.set(xlabel="$D_a$", ylabel=r"$\lambda_{\mathrm{comp}}$", ylim=(2e-3, 3.0))
    ax.set_title(r"(b) raising $D_a$ alone hits a ceiling set by $D_b$",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    print(f"  abar = bbar = {abar:.4f}; ceilings as D_a -> inf:")
    for Db_v in (100.0, 400.0, 1600.0):
        print(f"    D_b={Db_v:>6.0f}: exact {float(lam_exact(np.array(1e8), Db_v)):.4f}"
              f"   closed form {2 * alpha * abar**3 * Db_v / (3 * ell):.4f}")
    fig.tight_layout()
    _save(fig, "AB_tworeservoir_plane.png")


# =========================================================================== #
#  Figure 12(a-c) : the sign of V' decides whether co-located puncta sort     #
# =========================================================================== #
# sdot = 2 chi(s) V' c_max sech^2(s/2) tanh(s/2) with chi > 0 throughout, so
# sign(sdot) = sign(V'): s = 0 is the pair's only equilibrium and its eigenvalue
# lambda_pol = (8/7) V' c_max crosses zero linearly at V' = 0. ETDRK4 on the even
# extension onto [0, 2*ell], so the restriction to [0, ell] is the zero-flux
# problem exactly. Trajectories are cached under <outdir>/.cache: the fifteen
# integrations take ~40 min in total.
# Symmetric about zero and including V' = 0, which is the control: there the
# orientations decouple and the pair must sit still.
LADDER = (-0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03)
VS_LAMBDA = (0.01, 0.02, 0.03, -0.01, -0.02, -0.03)

# chi -> 8/7 only as s -> 0, so the rate is read near contact; the fitted slope
# is insensitive to the window (0.872 for (0.3, 0.6) against 0.865 here).
WIN = (0.45, 0.75)
T_SETTLE = 1.0          # seed relaxation transient, discarded


def fig12_sorting():
    _, _, _, _, solve_pde = _require_funpy()
    alpha, ell, n, D = ALPHA_SPIKE, ELL_SPIKE, 2.5, 1000.0
    N, chi0 = 512, 8.0 / 7.0
    abar = spike_reservoir(alpha, ell, n, K=1)
    cmax = 3.0 / (2 * alpha * abar**2)
    cache = os.path.join(OUTDIR, ".cache")
    os.makedirs(cache, exist_ok=True)
    x = (2.0 * ell) * np.arange(N) / N

    def run(Vp, s0, t_span, save_every, tag):
        path = os.path.join(cache, f"{tag}_V{Vp:+.5f}_s{s0:g}_T{t_span:g}.npz")
        if os.path.exists(path):
            return dict(np.load(path))
        sol = solve_pde(sorting_pde(alpha, D, 2.0 * ell, Vp, N),
                        sorting_seed(ell, n, N, ell / 2 - s0 / 2, ell / 2 + s0 / 2, cmax),
                        t_span=t_span, dt=0.002, N=N, adaptive=True,
                        rtol=1e-9, atol=1e-9, save_every=save_every)
        times = np.asarray(sol.times)
        ch, cdh = sol.raw(1), sol.raw(3)
        h = N // 2
        out = dict(
            times=times,
            sep=np.array([abs(_peak_centre(cdh[i], x, ell) - _peak_centre(ch[i], x, ell))
                          for i in range(len(times))]),
            pk=np.array([[ch[i][:h + 1].max(), cdh[i][:h + 1].max()]
                         for i in range(len(times))]),
            xh=x[:h + 1], cf=ch[-1][:h + 1], cdf=cdh[-1][:h + 1])
        np.savez(path, **out)
        return out

    data = {}
    for Vp in LADDER:
        data[Vp] = run(Vp, 1.0, 40.0, 20, "story")
        print(f"  V'={Vp:+.3f}: s {data[Vp]['sep'][0]:.3f} -> {data[Vp]['sep'][-1]:.3f}",
              flush=True)

    fig = plt.figure(figsize=(9.8, 7.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
    gsa = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                           hspace=0.14)

    # ---- (a) the outcome ----
    for row, (Vp, tag) in enumerate(((-0.03, "locked"), (0.03, "sorted"))):
        ax = fig.add_subplot(gsa[row])
        d = data[Vp]
        ax.plot(d["xh"], d["cf"], color="#4477AA", lw=1.8, label="$c$")
        ax.plot(d["xh"], d["cdf"], color="#CC6677", lw=1.8, ls="--",
                label=r"$c^\dagger$")
        ax.set(xlim=(ell / 2 - 11, ell / 2 + 11), ylim=(0, 24), yticks=[0, 20])
        ax.text(0.03, 0.88, rf"$\mathcal{{V}}'={Vp:+.2f}$  ({tag})",
                transform=ax.transAxes, fontsize=8)
        ax.set_ylabel("concentration", fontsize=8)
        if row == 0:
            ax.set_xticklabels([])
            ax.set_title("(a) the outcome", fontsize=9.5)
            ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.3)
        else:
            ax.set_xlabel("$x$")

    # ---- (b) the switch ----
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("coolwarm")
    vmax = max(abs(v) for v in LADDER)
    for Vp in LADDER:
        col = "0.45" if Vp == 0 else cmap(0.5 + 0.5 * Vp / vmax)
        ax.plot(data[Vp]["times"], data[Vp]["sep"], color=col,
                lw=1.9 if Vp == 0 else 1.6, ls=":" if Vp == 0 else "-",
                zorder=3 if Vp == 0 else 2)
    for xx, yy, lab, col in ((8.0, 6.05, r"$\mathcal{V}'>0$", cmap(0.95)),
                             (32.0, 1.30, r"$\mathcal{V}'=0$", "0.45"),
                             (32.0, 0.40, r"$\mathcal{V}'<0$", cmap(0.05))):
        ax.text(xx, yy, lab, ha="left", va="center", fontsize=8.5, color=col)
    ax.set(xlim=(0, 40.0), xlabel="$t$", ylabel="separation $s$")
    ax.set_title(r"(b) the sign of $\mathcal{V}'$ is the switch", fontsize=9.5)

    # ---- (c) the rate ----
    # s = 0 is an equilibrium, so the window has to be traversed: V' > 0 starts
    # inside it and separates, V' < 0 starts outside and collapses through.
    ax = fig.add_subplot(gs[1, 0])
    pts = []
    for Vp in VS_LAMBDA:
        s0, tsp = (0.30, 30.0) if Vp > 0 else (1.60, 9.0)
        d = run(Vp, s0, tsp, 10, "rate")
        t, s, pk = d["times"], d["sep"], d["pk"]
        keep = (t > T_SETTLE) & (s > WIN[0]) & (s < WIN[1])
        if keep.sum() < 3:
            continue
        pts.append((Vp, float(np.median(np.gradient(s, t)[keep] / s[keep])),
                    float(np.median(pk[keep]))))
    pts.sort()
    if pts:
        vp = np.array([q[0] for q in pts]); lam = np.array([q[1] for q in pts])
        cmx = float(np.median([q[2] for q in pts]))
        ax.plot(vp, lam, "o", ms=6, color="#4477AA", mec="w", mew=0.7, zorder=5,
                label=r"measured ($D_a=10^3$)")
        vg = np.linspace(-0.035, 0.035, 3)
        ax.plot(vg, chi0 * cmx * vg, color="#222222", lw=2.0, zorder=4,
                label=r"$\lambda_{\rm pol}=\frac{8}{7}\mathcal{V}^\prime c_{\max}$")
        slope = float(vp @ lam / (vp @ vp))
        print(f"  c_max={cmx:.3f}, predicted slope={chi0 * cmx:.3f}, "
              f"fitted={slope:.3f} (ratio {slope / (chi0 * cmx):.3f})")
    ax.axhline(0, color="0.85", lw=0.7); ax.axvline(0, color="0.85", lw=0.7)
    ax.set(xlabel=r"$\mathcal{V}^\prime$", ylabel=r"$\dot s/s$ near contact")
    ax.set_title("(c) the rate, linear through the origin", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    # ---- (d) the drift law ----
    # The one panel of the figure integrated with the finite-difference IMEX
    # scheme (two coupled triplets) rather than spectrally.
    ax = fig.add_subplot(gs[1, 1])
    pred, meas = _drift_law_points()
    lim = max(np.abs(pred).max(), np.abs(meas).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, label="$y=x$")
    ax.scatter(pred, meas, s=45, color="#6a3d9a", zorder=3)
    ax.set(xlabel=r"predicted $-2.5\,\mathcal{V}'\,(c^\dagger)'(x_c)$",
           ylabel=r"measured drift $\dot x_c$")
    ax.set_title(r"(d) Drift law (varying $\mathcal{V}'$, separation)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    print(f"  drift-law fit slope={np.polyfit(pred, meas, 1)[0]:.3f} (expect ~1.0)")

    # tight_layout is incompatible with the nested gridspec of panel (a)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.075)
    _save(fig, "AB_sorting.png")


# =========================================================================== #
#  Figure 12(d) machinery : the drift law, measured with the IMEX scheme      #
# =========================================================================== #
# Six short two-triplet integrations across both signs of V' and three
# separations, each compared with the prediction -2.5 V' (c^dag)'(x_c) of the
# constant-gradient drift law. Finite-difference IMEX, unlike panels (a)-(c).
def _drift_law_points():
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
    a0, c0 = relax(ELL / 2); print(f"  c_max={c0.max():.2f}")

    def step(a, ad, c, cd, kappa):
        Vc, Vcd = 1 + kappa * cd, 1 + kappa * c
        c = luC.solve(c + dt * (alpha * c**2 * a**2 - Vc * c)); a = luM.solve(a + dt * (-alpha * c**2 * a**2 + Vc * c))
        cd = luC.solve(cd + dt * (alpha * cd**2 * ad**2 - Vcd * cd)); ad = luM.solve(ad + dt * (-alpha * cd**2 * ad**2 + Vcd * cd))
        return a, ad, c, cd

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
    return np.array(pred), np.array(meas)


# =========================================================================== #
#  Driver                                                                     #
# =========================================================================== #
# Selectable by figure number or by output label; the numbers are the
# manuscript's current ones. Figure 12 is a single four-panel figure: panels
# (a)-(c) integrated spectrally, panel (d) by the finite-difference drift law.
FIGURES = {
    "2": fig2_3_dispersion_nullcline, "3": fig2_3_dispersion_nullcline,
    "4": fig4_full_six_species,
    "5": fig5_pinned_front,
    "6": fig6_spike,
    "7": fig7_competition,
    "8": fig8_competition_saturation,
    "9": fig9_coarsening,
    "10": fig10_two_reservoir,
    "11": fig11_two_reservoir_plane,
    "12": fig12_sorting,
    # aliases: the label each function writes
    "AB_dispersion": fig2_3_dispersion_nullcline,
    "AB_nullcline": fig2_3_dispersion_nullcline,
    "AB_onset_numerics": fig4_full_six_species,
    "AB_pinning_maxwell": fig5_pinned_front,
    "AB_spike": fig6_spike,
    "AB_competition": fig7_competition,
    "AB_competition_saturation": fig8_competition_saturation,
    "AB_coarsening": fig9_coarsening,
    "AB_tworeservoir": fig10_two_reservoir,
    "AB_tworeservoir_plane": fig11_two_reservoir_plane,
    "AB_sorting": fig12_sorting,
}

# Default run order (the aliases above would otherwise duplicate every entry).
DEFAULT_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]


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
    requested = args.figs or DEFAULT_ORDER
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
