"""Theta-method / Picard finite-volume kernel for coupled radial diffusion.

The solved system is, per field ``i``::

    W_i(u) * du_i/dt = (1/r**p) * d/dr [ r**p * coef_i(u) * du_i/dr ]

    r = 0 : du_i/dr = 0
    r = R : -coef_i * du_i/dr = k_i * (u_i - env_i(t))

``W_i`` is the *volume-integrated* capacity, not a density: for a concentration
field it is the cell volume, for a temperature field it is ``v * rho * cp``.
Getting this wrong preserves conservation of the assembled system while changing
every magnitude, which is hard to notice downstream.

Discretisation choices that matter:

* The flux form is conservative by construction. Interior face fluxes appear in
  two rows with opposite signs and cancel; only the outer boundary term
  survives in the domain sum.
* Interface conductivities use the harmonic mean. It is positive whenever both
  node values are positive and keeps the face flux pointing against the nodal
  gradient even when adjacent conductivities differ by orders of magnitude.
* Nonlinear coefficients are updated by Picard iteration inside every time step
  and the iteration runs to a tolerance, so the second-order time accuracy of
  the underlying Crank-Nicolson step is not thrown away.
* Coupled fields are advanced with block Gauss-Seidel: each field is solved
  with the latest values of the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .geometry import Grid

__all__ = [
    "Field",
    "thomas",
    "face_coef",
    "solve_field",
    "solve_coupled",
    "solve_moving",
]

State = Dict[str, List[float]]


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def thomas(
    lower: Sequence[float],
    diag: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> List[float]:
    """Solve a tridiagonal system in O(n).

    ``lower[0]`` and ``upper[-1]`` are ignored. Inputs are not modified.
    """
    n = len(diag)
    if not (len(lower) == len(upper) == len(rhs) == n):
        raise ValueError("thomas: all four arrays must have the same length")

    cp = [0.0] * n
    dp = [0.0] * n
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m

    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def face_coef(nodes: Sequence[float], mode: str = "harmonic") -> List[float]:
    """Map node conductivities to interior face conductivities.

    ``harmonic`` (default) is the series-resistance average ``2ab/(a+b)``;
    ``arithmetic`` is ``(a+b)/2``. Harmonic is the safer default for strongly
    varying coefficients.
    """
    n = len(nodes) - 1
    if mode == "arithmetic":
        return [0.5 * (nodes[i] + nodes[i + 1]) for i in range(n)]
    if mode != "harmonic":
        raise ValueError("face_coef: mode must be 'harmonic' or 'arithmetic'")
    out = [0.0] * n
    for i in range(n):
        a, b = nodes[i], nodes[i + 1]
        s = a + b
        out[i] = 0.0 if s == 0.0 else 2.0 * a * b / s
    return out


# ---------------------------------------------------------------------------
# Field description
# ---------------------------------------------------------------------------
@dataclass
class Field:
    """One diffusing scalar field.

    Parameters
    ----------
    name:
        Key used in the state dict passed to ``conductivity`` and ``capacity``.
    init:
        Scalar or length ``N+1`` sequence of initial values.
    conductivity:
        ``f(state) -> list`` of nodal conductivities (``D`` or ``k``).
    robin:
        Boundary conductance per unit area, scalar or ``f(t)``.
    env:
        ``f(t)`` ambient value used by the Robin condition.
    capacity:
        ``f(state) -> list``, a fixed sequence, or ``None`` for the cell volume.
        This is the volume-integrated capacity ``W_i``.
    reference:
        Reference level used by the ``"conservative"`` accounting mode.
    """

    name: str
    init: object
    conductivity: Callable[[State], Sequence[float]]
    robin: object
    env: Callable[[float], float]
    capacity: object = None
    reference: float = 0.0


def _init_array(init, n: int) -> List[float]:
    if isinstance(init, (int, float)):
        return [float(init)] * n
    arr = [float(x) for x in init]
    if len(arr) != n:
        raise ValueError("initial field has %d values, expected %d" % (len(arr), n))
    return arr


def _scalar(v, t: float) -> float:
    return float(v(t)) if callable(v) else float(v)


def _capacity(field: Field, grid: Grid, state: State) -> List[float]:
    cap = field.capacity
    if cap is None:
        return list(grid.v)
    if callable(cap):
        out = list(cap(state))
        if len(out) != grid.N + 1:
            raise ValueError("capacity for %r returned %d values" % (field.name, len(out)))
        return out
    return [float(x) for x in cap]


def _conductance(grid: Grid, coef: Sequence[float], mode: str, scale_k: float) -> List[float]:
    """Interior face conductances ``K_i = kf_i * a_i / (dr * scale_k)``.

    ``scale_k`` is the geometric factor from a coordinate change; it equals 1 on
    a fixed physical grid and ``R(t)**2`` in material coordinates.
    """
    kf = face_coef(coef, mode)
    dr = grid.dr
    a = grid.a
    inv = 1.0 / (dr * scale_k)
    return [kf[i] * a[i] * inv for i in range(grid.N)]


def _explicit_flux(
    grid: Grid, u: Sequence[float], K: Sequence[float], G: float, env: float
) -> List[float]:
    """Net outflow from each cell at the current time level."""
    N = grid.N
    out = [0.0] * (N + 1)
    out[0] = K[0] * (u[0] - u[1])
    for i in range(1, N):
        out[i] = K[i] * (u[i] - u[i + 1]) + K[i - 1] * (u[i] - u[i - 1])
    out[N] = K[N - 1] * (u[N] - u[N - 1]) + G * (u[N] - env)
    return out


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------
def solve_coupled(
    grid: Grid,
    fields: Sequence[Field],
    dt: float,
    nsteps: int,
    theta: float = 0.5,
    tol: float = 1e-10,
    max_sweeps: int = 30,
    interface: str = "harmonic",
    record: Optional[Sequence[int]] = None,
    probe: Optional[Callable[[int, State], None]] = None,
    scale_k: Optional[Callable[[float], float]] = None,
    scale_g: Optional[Callable[[float], float]] = None,
):
    """Advance several coupled fields over ``nsteps`` steps of size ``dt``.

    Parameters
    ----------
    theta:
        1.0 backward Euler, 0.5 Crank-Nicolson. Values in between are allowed.
    tol, max_sweeps:
        Picard convergence threshold on the max nodal change, and its cap.
    record:
        Node indices to keep in history. ``None`` keeps every node; pass an
        empty sequence to keep none.
    probe:
        ``f(step, state)`` called after every accepted step. Useful for early
        termination and for diagnostics that would be wasteful to store.
    scale_k, scale_g:
        Optional ``f(t)`` geometric factors. Default 1.

    Returns
    -------
    (history, stats)
        ``history`` maps field name to a list of sampled states (first entry is
        the initial state). ``stats`` holds iteration counts, residuals and the
        accumulated accounting terms described in ``docs/methodology.md``.
    """
    if not fields:
        raise ValueError("at least one field is required")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must lie in (0, 1]")

    N = grid.N
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ValueError("field names must be unique")

    _scale_k = scale_k or (lambda t: 1.0)
    _scale_g = scale_g or (lambda t: 1.0)

    state: State = {f.name: _init_array(f.init, N + 1) for f in fields}

    if record is None:
        idx = list(range(N + 1))
    else:
        idx = [int(i) for i in record]
        for i in idx:
            if not (0 <= i <= N):
                raise ValueError("record index %d outside 0..%d" % (i, N))

    history = {f.name: [[state[f.name][i] for i in idx]] for f in fields}

    acc = {
        f.name: {"flux": 0.0, "exact": 0.0, "conservative": 0.0, "engineering": 0.0}
        for f in fields
    }
    sweeps_used = []
    unconverged = 0

    env_cache = {f.name: [] for f in fields}
    for n in range(nsteps + 1):
        t = n * dt
        for f in fields:
            env_cache[f.name].append(float(f.env(t)))

    for n in range(nsteps):
        t0 = n * dt
        t1 = (n + 1) * dt

        old = {k: list(v) for k, v in state.items()}
        W_old = {f.name: _capacity(f, grid, old) for f in fields}

        k_old = _scale_k(t0)
        g_old = _scale_g(t0)
        k_new = _scale_k(t1)
        g_new = _scale_g(t1)

        # Robin conductances depend only on time, not on the state, so they are
        # evaluated once per time level instead of once per Picard sweep.
        F_old = {}
        G_old = {}
        G_new = {}
        for f in fields:
            coef_old = [float(x) for x in f.conductivity(old)]
            K = _conductance(grid, coef_old, interface, k_old)
            G_old[f.name] = _scalar(f.robin, t0) * grid.a_R / g_old
            G_new[f.name] = _scalar(f.robin, t1) * grid.a_R / g_new
            F_old[f.name] = _explicit_flux(
                grid, old[f.name], K, G_old[f.name], env_cache[f.name][n]
            )

        used = 0
        converged = False
        for m in range(max_sweeps):
            used = m + 1
            worst = 0.0
            for f in fields:
                coef = [float(x) for x in f.conductivity(state)]
                W = _capacity(f, grid, state)
                K = _conductance(grid, coef, interface, k_new)
                G = G_new[f.name]

                lo = [0.0] * (N + 1)
                dg = [0.0] * (N + 1)
                up = [0.0] * (N + 1)
                rh = [0.0] * (N + 1)

                dg[0] = W[0] + dt * theta * K[0]
                up[0] = -dt * theta * K[0]
                for i in range(1, N):
                    lo[i] = -dt * theta * K[i - 1]
                    dg[i] = W[i] + dt * theta * (K[i] + K[i - 1])
                    up[i] = -dt * theta * K[i]
                lo[N] = -dt * theta * K[N - 1]
                dg[N] = W[N] + dt * theta * (K[N - 1] + G)

                fo = F_old[f.name]
                uo = old[f.name]
                cf = dt * (1.0 - theta)
                for i in range(N + 1):
                    rh[i] = W[i] * uo[i] - cf * fo[i]
                rh[N] += dt * theta * G * env_cache[f.name][n + 1]

                sol = thomas(lo, dg, up, rh)
                prev = state[f.name]
                delta = 0.0
                for i in range(N + 1):
                    d = sol[i] - prev[i]
                    if d < 0.0:
                        d = -d
                    if d > delta:
                        delta = d
                state[f.name] = sol
                if delta > worst:
                    worst = delta

            if worst < tol:
                converged = True
                break

        if not converged:
            unconverged += 1
        sweeps_used.append(used)

        # ---- accounting (see docs/methodology.md) --------------------------
        for f in fields:
            W_new = _capacity(f, grid, state)
            u_new = state[f.name]
            u_old = old[f.name]
            ref = float(f.reference)

            g_n = G_old[f.name]
            g_p = G_new[f.name]
            flux = dt * (
                theta * g_p * (env_cache[f.name][n + 1] - u_new[N])
                + (1.0 - theta) * g_n * (env_cache[f.name][n] - u_old[N])
            )

            a = acc[f.name]
            a["flux"] += flux
            for i in range(N + 1):
                a["exact"] += W_new[i] * (u_new[i] - u_old[i])
                a["conservative"] += W_new[i] * (u_new[i] - ref) - W_old[f.name][i] * (
                    u_old[i] - ref
                )
                a["engineering"] += W_new[i] * u_new[i] - W_old[f.name][i] * u_old[i]

        for f in fields:
            history[f.name].append([state[f.name][i] for i in idx])

        if probe is not None:
            probe(n + 1, state)

    stats = {
        "sweeps_max": max(sweeps_used) if sweeps_used else 0,
        "sweeps_mean": (sum(sweeps_used) / len(sweeps_used)) if sweeps_used else 0.0,
        "unconverged_steps": unconverged,
        "record": idx,
        "fields": {},
    }
    for f in fields:
        a = acc[f.name]
        denom = abs(a["flux"]) if a["flux"] != 0.0 else 1.0
        stats["fields"][f.name] = {
            "flux": a["flux"],
            "storage_exact": a["exact"],
            "storage_conservative": a["conservative"],
            "storage_engineering": a["engineering"],
            "residual": abs(a["exact"] - a["flux"]) / denom,
            "gap_conservative": (a["conservative"] - a["exact"]) / denom,
            "gap_engineering": (a["engineering"] - a["exact"]) / denom,
        }
    return history, stats


def solve_field(
    grid: Grid,
    u0,
    dt: float,
    nsteps: int,
    conductivity: Callable[[Sequence[float]], Sequence[float]],
    robin: object,
    env: Callable[[float], float],
    capacity=None,
    name: str = "u",
    **kwargs,
):
    """Single-field convenience wrapper around :func:`solve_coupled`.

    ``conductivity`` takes the nodal array directly rather than the state dict.
    Returns ``(history_list, stats)`` where ``history_list`` is the list of
    sampled arrays for this field.
    """
    field = Field(
        name=name,
        init=u0,
        conductivity=lambda s: conductivity(s[name]),
        robin=robin,
        env=env,
        capacity=capacity,
    )
    history, stats = solve_coupled(grid, [field], dt, nsteps, **kwargs)
    return history[name], stats


def solve_moving(
    grid: Grid,
    fields: Sequence[Field],
    dt: float,
    nsteps: int,
    radius: Callable[[float], float],
    **kwargs,
):
    """Coupled solve in material coordinates ``xi = r / R(t)``.

    ``grid`` must be built on the normalised domain, i.e. ``Grid(1.0, N, geom)``.
    A uniformly shrinking body carries the moisture with the skeleton, so the
    advection term and the ``1/R**2`` capacity term cancel exactly and the
    transformed equation keeps the diffusion form:

        du/dt = (1 / (R(t)**2 * xi**p)) * d/dxi [ xi**p * coef * du/dxi ]

    The two geometric scalings are not symmetric. Interior conductances scale as
    ``1/R**2`` while the boundary conductance scales as ``1/R``, because the
    surface area shrinks like ``R`` while the volume shrinks like ``R**2``.
    Using ``1/R**2`` in both places understates the boundary resistance.

    Returns ``(history, stats)`` with ``stats["R"]`` holding the radius at every
    recorded step.
    """
    if abs(grid.R - 1.0) > 1e-12:
        raise ValueError("solve_moving requires a grid on [0, 1] (material coordinates)")

    R_hist = [float(radius(n * dt)) for n in range(nsteps + 1)]

    history, stats = solve_coupled(
        grid,
        fields,
        dt,
        nsteps,
        scale_k=lambda t: float(radius(t)) ** 2,
        scale_g=lambda t: float(radius(t)),
        **kwargs,
    )
    stats["R"] = R_hist
    return history, stats
