# puncta

Code to reproduce every figure in

> Hossein, Alwani, Buttenschön & Fletcher,
> "Clustering versus sorting: a mass-conserving reaction–diffusion model
> of planar polarity puncta" (submitted).

The single script `puncta_modelling_manuscript.py` generates all results
figures; run `python3 puncta_modelling_manuscript.py` for everything, or
pass figure numbers (e.g. `python3 puncta_modelling_manuscript.py 2 3 7`).
Output is written to `figures/`. Per-figure grid sizes, time steps and
integration times are set in the script; physical parameters are those of
Table 1 of the manuscript.

## Dependencies

Python 3.10–3.13 with NumPy, SciPy and Matplotlib throughout. Figures 8,
11 and 12(a)–(c) additionally require the open-source
[funpy](https://github.com/adrs0049/funpy) spectral library, which
supplies the ultraspherical resolvent behind the competition eigenvalue
problems and the ETDRK4 integrator behind the sorting runs:

    git clone https://github.com/adrs0049/funpy
    cd funpy && pip install .

funpy is imported only when one of those three figures is requested, so
every other figure runs without it.

## A note for Windows users

funpy builds Cython/C extensions on installation, which on Windows
requires the Microsoft C++ compiler ("Microsoft C++ Build Tools", with
the *Desktop development with C++* workload). If `pip install .` fails
with `Microsoft Visual C++ 14.0 or greater is required`, either install
those build tools, or — more simply — install and run funpy under
[WSL](https://learn.microsoft.com/windows/wsl/), where
`sudo apt install build-essential python3-dev` is sufficient. All
figures were generated under Linux.

## Runtimes

Most figures take seconds to a few minutes. The sorting runs of
Figure 12(a)–(c) take roughly 40 minutes in total on a laptop; the
trajectories are cached under `figures/.cache`, so re-running is fast.
The measured competition rates in Figure 8(a) are tabulated in the
script, having been computed with the diagnostic of Figure 7(b) (the
slowest of those integrations alone takes ~45 minutes); the tabulating
code path is `fig7_competition`.
