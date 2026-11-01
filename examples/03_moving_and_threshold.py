"""Shrinking cylinder in material coordinates, and a threshold time.

Two things are demonstrated:

1. ``solve_moving`` on a radius that changes with time. The coordinate is
   xi = r / R(t), so the domain stays [0, 1] and the grid never needs rebuilding.
   Interior conductances scale as 1/R^2 and the boundary conductance as 1/R.
2. ``crossing_time`` turning a sampled decay into a continuous instant.

    python examples/03_moving_and_threshold.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvmkit import Field, Grid, solve_moving
from fvmkit.envtab import blend_to_hold, crossing_time, pchip

R0 = 0.02
N = 60
DT = 4.0
T_END = 7200.0

# radius table: shrinks by about 12% over the window, then holds
T_R = [0.0, 1800.0, 3600.0, 5400.0]
R_TAB = [R0, 0.0192, 0.0183, 0.0176]
radius = blend_to_hold(pchip(T_R, R_TAB), T_R[-1], R_TAB[-1], taper=900.0)

# ambient, likewise tabulated
T_AMB_T = [0.0, 1200.0, 2400.0]
T_AMB = [300.15, 330.15, 340.15]
C_AMB = [0.11, 0.095, 0.088]
t_air = blend_to_hold(pchip(T_AMB_T, T_AMB), T_AMB_T[-1], T_AMB[-1], taper=600.0)
c_env = blend_to_hold(pchip(T_AMB_T, C_AMB), T_AMB_T[-1], C_AMB[-1], taper=600.0)

D0 = 6.0e-6
E_OVER_R = 1800.0
K0 = 0.35
RHO_CP = 2.4e6


def main():
    grid = Grid(1.0, N, "cylinder")          # material coordinates
    fields = [
        Field(
            "c",
            2.2,
            lambda s: [D0 * math.exp(-E_OVER_R / s["T"][i]) for i in range(grid.N + 1)],
            9.0e-7,
            c_env,
        ),
        Field(
            "T",
            300.15,
            lambda s: [K0] * (grid.N + 1),
            24.0,
            t_air,
            capacity=[grid.v[i] * RHO_CP for i in range(grid.N + 1)],
        ),
    ]

    history, stats = solve_moving(grid, fields, DT, int(T_END / DT), radius)

    print("material grid : N=%d, dxi=%.4f" % (grid.N, 1.0 / grid.N))
    print("radius        : %.4f cm -> %.4f cm" % (stats["R"][0] * 100, stats["R"][-1] * 100))
    print("sweeps        : max %d, mean %.2f" % (stats["sweeps_max"], stats["sweeps_mean"]))
    for name, s in stats["fields"].items():
        print("  %-2s residual %.2e" % (name, s["residual"]))

    ts = [n * DT for n in range(int(T_END / DT) + 1)]
    centre = [row[0] for row in history["c"]]
    print("\n  centre moisture %.4f -> %.4f" % (centre[0], centre[-1]))

    t_cross, info = crossing_time(ts, centre, 1.5)
    if t_cross is None:
        print("  no crossing of 1.50 in the window")
        return
    print("  crosses 1.50 at t = %.3f s" % t_cross)
    print("  sampling bracket  [%.1f, %.1f] s" % info["bracket"])
    print("  straight-line reading differs by %+.3f ms" % (-info["linear_gap"] * 1e3))
    print("  interpolant residual at the root %.2e" % info["residual"])


if __name__ == "__main__":
    main()
