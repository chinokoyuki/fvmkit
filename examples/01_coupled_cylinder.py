"""Coupled heat and moisture diffusion in a cylinder with a tabulated ambient.

Two fields, both nonlinear through the same closure:

    dC/dt       = (1/r) d/dr [ r D(C,T) dC/dr ]
    rho(C)cp(C) dT/dt = (1/r) d/dr [ r k(C) dT/dr ]

    r = R :  -D dC/dr = h_m (C_s - C_e(t))
             -k dT/dr = h   (T_s - T_air(t))

The ambient curves come from a short table and are extended past its end with a
C1 blend into a hold value, so the boundary condition does not jump.

Run from the repository root:

    python examples/01_coupled_cylinder.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvmkit import Field, Grid, solve_coupled
from fvmkit.envtab import blend_to_hold, crossing_time, pchip

# ---- geometry and time -----------------------------------------------------
RADIUS = 0.02          # m
N = 80                 # cells; dr = 0.25 mm
DT = 8.0               # s
T_END = 14400.0        # s
T0 = 2000.0            # elapsed time of the last table entry

# ---- ambient table (t [s], T_air [K], C_e [kg/kg]) -------------------------
TIMES = [0.0, 300.0, 900.0, 1500.0, 2000.0]
T_AIR = [301.15, 320.15, 338.15, 343.15, 343.65]
C_ENV = [0.115, 0.105, 0.095, 0.090, 0.089]

T_HOLD = sum(T_AIR[-3:]) / 3.0
C_HOLD = sum(C_ENV[-3:]) / 3.0

t_air = blend_to_hold(pchip(TIMES, T_AIR), T0, T_HOLD, taper=600.0)
c_env = blend_to_hold(pchip(TIMES, C_ENV), T0, C_HOLD, taper=600.0)

# ---- property closure ------------------------------------------------------
C_INIT = 2.0
T_INIT = 301.15


def props(state):
    """Return rho, cp, k, D for the current state.

    The diffusivity uses an Arrhenius factor, so temperature must be in kelvin.
    Mixing celsius in here changes D by several orders of magnitude without
    raising anything.
    """
    c = state["c"]
    t = state["T"]
    rho = [800.0 + 120.0 * c[i] for i in range(len(c))]
    cp = [1500.0 + 2400.0 * c[i] / (c[i] + 1.0) for i in range(len(c))]
    k = [0.30 + 0.15 * c[i] / (c[i] + 1.0) for i in range(len(c))]
    d = [5.0e-7 * math.exp(-0.45 / max(c[i], 1e-6)) * math.exp(-1200.0 / t[i]) for i in range(len(c))]
    return rho, cp, k, d


def fields_for(grid):
    def moisture_coef(state):
        return props(state)[3]

    def heat_coef(state):
        return props(state)[2]

    def heat_capacity(state):
        rho, cp = props(state)[0], props(state)[1]
        return [grid.v[i] * rho[i] * cp[i] for i in range(grid.N + 1)]

    return [
        Field(
            name="c",
            init=C_INIT,
            conductivity=moisture_coef,
            robin=8.0e-7,
            env=c_env,
            reference=C_HOLD,
        ),
        Field(
            name="T",
            init=T_INIT,
            conductivity=heat_coef,
            robin=25.0,
            env=t_air,
            capacity=heat_capacity,
            reference=273.15,
        ),
    ]


def main():
    grid = Grid(RADIUS, N, "cylinder")
    fields = fields_for(grid)
    nsteps = int(round(T_END / DT))

    history, stats = solve_coupled(grid, fields, DT, nsteps, theta=0.5, tol=1e-10)

    print("grid      : N=%d  dr=%.4f mm  %s" % (grid.N, grid.dr * 1e3, grid.geom))
    print("steps     : %d x %.1f s" % (nsteps, DT))
    print("picard    : max %d, mean %.2f, unconverged %d"
          % (stats["sweeps_max"], stats["sweeps_mean"], stats["unconverged_steps"]))
    print()
    print("accounting (relative to the integrated boundary flux):")
    for name, s in stats["fields"].items():
        print("  %-2s residual %.2e   conservative gap %+.2e   engineering gap %+.2e"
              % (name, s["residual"], s["gap_conservative"], s["gap_engineering"]))
    print("  residual is the discrete identity; the gaps measure bookkeeping")
    print("  conventions, not solver error. See docs/methodology.md.")

    print()
    print("t = %.0f s" % T_END)
    print("  C  centre %.4f   surface %.4f   mean %.4f"
          % (history["c"][-1][0], history["c"][-1][-1], grid.mean(history["c"][-1])))
    print("  T  centre %.2f   surface %.2f   mean %.2f  [K]"
          % (history["T"][-1][0], history["T"][-1][-1], grid.mean(history["T"][-1])))

    # Centre moisture over time, then the instant it passes a level.
    ts = [n * DT for n in range(nsteps + 1)]
    centre = [row[0] for row in history["c"]]
    t_cross, info = crossing_time(ts, centre, 1.5)
    if t_cross is None:
        print("  centre never reaches 1.50 within the window")
    else:
        print("  centre passes 1.50 at t = %.2f s  (between samples %.0f and %.0f s)"
              % (t_cross, info["bracket"][0], info["bracket"][1]))

    deltas = [abs(centre[i + 1] - centre[i]) for i in range(len(centre) - 1)]
    print("  largest one-step change in centre moisture: %.3e" % max(deltas))


if __name__ == "__main__":
    main()
