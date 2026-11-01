"""Run the three verification checks and print what each one measures.

    python examples/02_verification.py

Check 1  analytic series       - does the discrete solution approach the closed
                                 form of a sibling problem with frozen properties?
Check 2  observed order        - does the error fall by 4x per 2x refinement?
Check 3  conservation residual - does the discrete budget close on the
                                 production run, not just on a toy case?

A scheme can pass any one of these and still be wrong. The three fail
independently, which is the reason to run all three.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvmkit import Grid, solve_field
from fvmkit.verify import convergence_orders, self_convergence, series_solution

RADIUS = 0.02
ALPHA = 1.0e-9
ROBIN = 5.0e-7          # Bi = robin * R / alpha = 10
LEVEL_T = 900.0


def fixed_run(geom, n, dt=1.0):
    grid = Grid(RADIUS, n, geom)
    hist, stats = solve_field(
        grid,
        [1.0] * (grid.N + 1),
        dt,
        int(LEVEL_T / dt),
        lambda u: [ALPHA] * (grid.N + 1),
        ROBIN,
        lambda t: 0.0,
    )
    return grid.mean(hist[-1]), stats


def main():
    print("reference: constant-property sibling problem, uniform initial state,")
    print("Robin boundary, Bi = %.1f\n" % (ROBIN * RADIUS / ALPHA))

    for geom in ("slab", "cylinder", "sphere"):
        ref = series_solution(geom, RADIUS, ALPHA, ROBIN, 1.0, 0.0, n_modes=3000)
        target = ref.mean(LEVEL_T)

        values, last_stats = [], None
        for n in (40, 80, 160):
            v, s = fixed_run(geom, n)
            values.append(v)
            last_stats = s

        res = convergence_orders(values, target, ratio=2.0)
        print("%-9s analytic %.10f" % (geom, target))
        for i, (v, e) in enumerate(zip(values, res["errors"])):
            print("          N=%3d   %.10f   error %.2e" % ((40, 80, 160)[i], v, e))
        print("          observed order %.3f   (2.0 expected for central fluxes + CN)"
              % res["orders"][-1])
        print("          conservation residual %.2e" % last_stats["fields"]["u"]["residual"])
        print()

    # Time refinement measured against itself, so the fixed spatial error
    # cancels: only consecutive differences are used.
    grid = Grid(RADIUS, 80, "cylinder")
    values = []
    for nsteps in (25, 50, 100, 200):
        hist, _ = solve_field(
            grid,
            [1.0] * (grid.N + 1),
            400.0 / nsteps,
            nsteps,
            lambda u: [ALPHA] * (grid.N + 1),
            ROBIN,
            lambda t: 0.0,
        )
        values.append(grid.mean(hist[-1]))
    res = self_convergence(values, ratio=2.0)
    print("time refinement at fixed N=80")
    print("  successive differences %s"
          % "  ".join("%.3e" % d for d in res["differences"]))
    print("  observed orders        %s" % "  ".join("%.3f" % o for o in res["orders"]))


if __name__ == "__main__":
    main()
