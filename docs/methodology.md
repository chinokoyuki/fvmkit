# Verification protocol

A solver that produces smooth output is not evidence that it solves the intended
equation. This document describes the three checks `fvmkit.verify` supports, what
each one can and cannot detect, and the order to run them in.

The short version: run all three, on the production configuration, before
quoting any number.

## Why three

| check | question it answers | what it misses |
|---|---|---|
| closed-form reference | is the discrete operator the right operator? | errors that leave the field unchanged but break the budget |
| observed order | does the error fall at the design rate? | a reference that is consistently wrong |
| conservation residual | does the discrete budget close? | a wrong constitutive closure or a wrong accounting convention |

They fail independently. A sign error in a boundary condition passes the budget
check and fails the reference. A mis-accounted capacity passes the reference and
fails the budget. A first-order boundary treatment passes both at coarse
resolution and shows up only under refinement.

## 1. Closed-form reference

Freeze everything that can be frozen. Constant coefficient, uniform initial
state, constant ambient value, Robin boundary. The resulting problem has a
series solution, so the discretisation error can be isolated from every other
error.

For `-alpha du/dr = k_R (u - env)` at `r = R`, with `Bi = k_R R / alpha`, the
eigenvalues solve

| geometry | eigenvalue equation | eigenfunction |
|---|---|---|
| slab | `mu tan(mu) = Bi` | `cos(mu xi)` |
| cylinder | `mu J1(mu) = Bi J0(mu)` | `J0(mu xi)` |
| sphere | `tan(mu) = mu / (1 - Bi)` | `sin(mu xi) / xi` |

and the solution is `u = env + sum A_n psi_n(xi) exp(-mu_n^2 alpha t / R^2)`
with `xi = r / R`.

### Mode count

With a step initial condition and a large Biot number, the coefficients decay
slowly and alternate in sign. Twenty or thirty modes leave a truncation error of
order 1 near `t = 0`. Four hundred modes bring it below `1e-4` over the usual
inspection window.

The rule: choose `n_modes` so the truncation error is at least two orders of
magnitude below the discretisation error you intend to measure. If the measured
error stops shrinking under refinement, suspect the mode count before the
scheme.

### Sign of the offset

The driving offset is `theta_0 = u_init - u_env`, and it is signed. A body that
starts colder than its surroundings has `theta_0 < 0`, and every coefficient
flips with it. Taking `abs()` at this point produces a reference that is smooth,
plausible, and wrong in a way that only a convergence study will reveal.

### Point values versus volume averages

The pointwise series at the axis converges slowly, because every eigenfunction
tends to a constant there and the partial sums oscillate. Volume averages
converge much faster, since the mode weights decay like `mu^-2` rather than
`mu^-1`.

Use volume averages for tight tolerances. If a point value is required, raise the
mode count and check that the answer is stable in `n_modes` before using it.

## 2. Observed order

Two estimators, for two situations.

**Against the reference.** Given errors `e1, e2, e3` on grids refined by a factor
of two, `p = log2(e1 / e2)`. `fvmkit.verify.convergence_orders` returns these.
Requires a trustworthy reference.

**Against itself.** Refining one axis while holding the other fixed leaves the
error on the fixed axis unchanged, so the differences between successive
solutions isolate the axis being refined. `fvmkit.verify.self_convergence`
returns `p = log2(|u1-u2| / |u2-u3|)`. This is the practical way to measure the
temporal order, since no closed form is needed.

Readings and what they usually mean:

| observed order | reading |
|---|---|
| 1.9 – 2.1 | second-order as designed |
| 1.0 – 1.2 | a first-order slip; check the boundary row and the face averaging |
| 0.0 – 0.5 | differences dominated by the axis you are not refining |
| negative | diverging; look for a sign error or an unstable coupling |
| error flat while refining | the reference is wrong, or the feature is under-resolved |

The last row is the one that wastes the most time. An error ratio pinned near
1.00 is the signature of a wrong analytic setup far more often than a broken
scheme.

## 3. Conservation

The discrete identity for one step, with `G` the outer boundary conductance and
`W` the volume-integrated capacity, is

```
sum_i W_i^new (u_i^new - u_i^old)
    = dt [ theta G^new (env^new - u_N^new) + (1-theta) G^old (env^old - u_N^old) ]
```

Interior face fluxes cancel in the sum. Only the boundary term survives. That is
what "conservative" means here, and it holds for any `theta` in `(0, 1]`.

The trap is that a report may accumulate something else. Two common alternatives:

| convention | accumulated quantity | difference from the identity |
|---|---|---|
| reference-based | `sum [W_new(u_new-ref) - W_old(u_old-ref)]` | `sum (W_new - W_old)(ref - u_old)` |
| engineering | `sum (W_new u_new - W_old u_old)` | `sum (W_new - W_old) u_old` |

Neither difference shrinks under refinement when the capacity changes. Over a
process where `rho cp` varies by tens of percent, both are of order 1. A residual
of order 1 therefore does not show that the scheme leaks; it usually shows that
the audit and the solver are measuring different functionals.

`fvmkit.kernel` reports all three. `residual` is the identity above and should
reach machine precision. `gap_conservative` and `gap_engineering` report the
distance to the other two conventions, and are meant to be explained rather than
minimised.

When the capacity is constant, the three coincide. A concentration field with no
density change is the clean case to test on; if its residual is not near `1e-13`,
something is genuinely wrong.

Two further rules:

- Report the residual for the **production** run. A small case can be
  conservative through symmetry while the real configuration is not.
- A residual that stalls at `1e-5` rather than `1e-13` usually means the capacity
  used in the accounting differs from the one used in the assembly.

## 4. Order of operations

1. Series check on all three geometries at coarse resolution. This catches sign
   errors and axis handling.
2. Refine space. Confirm the order is near 2.
3. Refine time separately, using `self_convergence`.
4. Run the production configuration. Read the residual.
5. Only now look at the physical answer.

Steps 1–4 do not depend on the answer being physically plausible, which is why
they come first. Once the answer is on screen it is hard to evaluate it
neutrally.

## 5. Failure signatures

| symptom | likely cause |
|---|---|
| surface value oscillates in time | time step too large for the boundary flux, or Robin term divided by the wrong volume |
| axis value drifts when refinement changes | the half cell is missing, or the axis coefficient is not `2(p+1)/dr^2` |
| total mass grows slowly | interior fluxes not antisymmetric; check the face index in both rows |
| residual `O(1)` but field looks right | accounting convention mismatch, see above |
| order 1 at the boundary only | one-sided boundary treatment instead of the implicit Robin row |
| order 2 in space, 1 in time | Picard iteration stopped early, so the implicit step is effectively lagged |
| answer depends on `n_modes` in the reference | not enough modes; raise it before drawing conclusions |
| moving boundary result differs from fixed | boundary conductance scaled as `1/R^2` instead of `1/R` |

## 6. Modelling-choice audit

A verified solver still answers the wrong question if a closure was chosen badly.
Enumerate the choices the problem leaves open, and quantify each one's effect on
the headline number.

| choice | how to bound it |
|---|---|
| energy equation closure | solve both forms; the difference is `(W_new - W_old)(T_ref - T^n)` |
| latent heat or evaporation | operator-split an effective boundary value; check whether the surface crosses below the ambient |
| equilibrium value | sweep it; note whether the answer depends on it in the reported window |
| transfer coefficients | sweep by a factor of 2 to 10; usually dominant |
| omitted cross-effects | bound analytically, or state the omission and its direction |

Two rules that resolve most disagreements:

If a result crosses a hard physical bound — a surface temperature above the
ambient, a concentration below equilibrium — the closure is wrong. The code is
probably fine.

If the answer conflicts with an independently known duration or scale, do not
tune a parameter to fit it. Identify which parameter would have to move, by how
much, and report the inconsistency. A named inconsistency is a stronger result
than a fitted constant.

## Checklist

- [ ] Closed-form comparison run on the geometry actually used.
- [ ] `n_modes` large enough that the reference is stable in `n_modes`.
- [ ] The offset sign in the reference is the physical one.
- [ ] Spatial order measured, not assumed.
- [ ] Temporal order measured separately, on a fixed grid.
- [ ] Conservation residual read from the production run.
- [ ] All three accounting conventions reported or consciously discarded.
- [ ] Modelling-choice table produced; the largest sensitivity named.
- [ ] Every quoted number traceable to an artifact, not to console output.
