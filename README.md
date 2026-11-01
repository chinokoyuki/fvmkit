# fvmkit

Conservative finite-volume kernels for transient radial diffusion, plus the
checks that decide whether the numbers coming out of them can be trusted.

The solver handles slabs, cylinders and spheres with state-dependent
properties, Robin boundaries driven by tabulated data, several coupled fields,
and optionally a shrinking domain in material coordinates. It is pure Python
with no required dependencies.

The verification side is the reason this exists as a package rather than a
script. It supplies a closed-form reference for a sibling problem with frozen
coefficients, an observed-order estimator, and a conservation audit that
separates a genuine budget failure from a bookkeeping mismatch.

```
                         solve                     prove
  tabulated env  -->  fvmkit.kernel  -->  fvmkit.verify
  properties     -->  (theta, Picard) -->  (series, order, residual)
```

## Contents

| module | what it provides |
|---|---|
| `fvmkit.geometry` | control volumes for `slab`, `cylinder`, `sphere` on `[0, R]` |
| `fvmkit.kernel` | theta method with Picard iteration, block Gauss-Seidel coupling, material coordinates |
| `fvmkit.envtab` | shape-preserving interpolation, C1 continuation of tables, continuous threshold times |
| `fvmkit.verify` | Bessel and trigonometric series solutions, Richardson extrapolation, observed order, self-convergence |
| `fvmkit.audit` | one-at-a-time sweeps, inverse calibration, accounting comparison, thin-layer fits |
| `fvmkit.docsync` | stale-number scanning and duplicate-artifact detection for generated reports |

## Install

```bash
pip install -e .
```

Optional extras: `pip install -e ".[reference]"` for scipy-backed Bessel
functions, `.[plot]` for matplotlib, `.[dev]` for pytest.

## Quick start

```python
from fvmkit import Field, Grid, solve_coupled

grid = Grid(R=0.02, N=80, geom="cylinder")

fields = [
    Field("c", init=2.0,
          conductivity=lambda s: [D_of(s["c"][i], s["T"][i]) for i in range(grid.N + 1)],
          robin=8e-7, env=lambda t: 0.09),
    Field("T", init=301.15,
          conductivity=lambda s: [k_of(s["c"][i]) for i in range(grid.N + 1)],
          robin=25.0, env=lambda t: 340.0,
          capacity=lambda s: [grid.v[i] * rho_cp(s["c"][i]) for i in range(grid.N + 1)]),
]

history, stats = solve_coupled(grid, fields, dt=8.0, nsteps=1800, theta=0.5)

print(stats["fields"]["c"]["residual"])   # ~1e-14 on a well-posed problem
```

`Field.capacity` is the volume-integrated capacity, not a density: cell volume
for a concentration, `v * rho * cp` for a temperature. Passing `rho * cp`
instead keeps the scheme conservative while changing every magnitude, which is
easy to miss.

## Verification

`examples/02_verification.py` runs the checks against a frozen-property problem
that has a closed-form solution. Measured on this machine:

| geometry | error at N=40 | N=80 | N=160 | observed order |
|---|---:|---:|---:|---:|
| slab | 1.97e-04 | 4.92e-05 | 1.23e-05 | 1.997 |
| cylinder | 3.88e-04 | 9.71e-05 | 2.43e-05 | 1.998 |
| sphere | 5.74e-04 | 1.44e-04 | 3.59e-05 | 1.998 |

Time refinement at fixed `N=80` gives successive differences of
`1.64e-07, 4.09e-08, 1.02e-08`, i.e. an observed order of 2.000. Conservation
residuals sit between 3e-14 and 2e-13 relative to the integrated boundary flux.

These are measurements of the current code, not targets. Run the script after
changing anything in the kernel.

## Accuracy and cost

Second order in space and time. The axis is handled by the half control volume,
which reproduces the L'Hopital coefficient `2(p+1)/dr^2` without a branch, so
there is no special case to get wrong at `r = 0`.

Cost is pure-Python. A 80-cell cylinder over 1800 steps takes about two seconds;
a 320-cell grid over 259 200 steps takes several minutes. If you need speed,
the kernel is small enough to port or to wrap with numpy at the inner loops,
but check the convergence numbers again afterwards.

## Scope

In scope: linear diffusion operators in one radial dimension, nonlinear
coefficients that depend on the solution, Robin and symmetry boundaries,
multiple coupled fields, and a prescribed radius history.

Out of scope and left to the caller: source terms, latent-heat or other
cross-coupling closures, anisotropic media, and moving boundaries whose
velocity is not uniform across the section. The material-coordinate transform
relies on a rigid, uniform shrinkage so that the advection term cancels the
capacity term; that assumption is stated explicitly in `solve_moving`.

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — the verification protocol, in
  full, including the failure signatures it is meant to catch.
- [`README.zh-CN.md`](README.zh-CN.md) — Chinese version of this file.

## License

MIT.
