"""Reference solutions and convergence diagnostics.

A solver that runs is not evidence that a solver is right. This module supplies
the three independent checks worth running before any number is reported:

1. a closed-form reference for a sibling problem with frozen coefficients,
2. the observed order of accuracy under refinement,
3. the conservation residual of the production run.

They fail in different ways, which is the point. A wrong reference solution, a
first-order slip in the time integration and a mis-accounted capacity all look
plausible on their own; they do not agree with each other.

Closed forms
------------
For constant ``alpha`` and a Robin condition ``-alpha du/dr = k_R (u - env)`` at
``r = R``, the transient solution is a series over eigenvalues of

    slab     : mu tan(mu)          = Bi
    cylinder : mu J1(mu)           = Bi J0(mu)
    sphere   : tan(mu)             = mu / (1 - Bi)

with ``Bi = k_R R / alpha``. The coefficients below are the projections of a
uniform initial condition onto each eigenfunction, already divided by the
eigenfunction norm.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

__all__ = [
    "bessel_j0",
    "bessel_j1",
    "robin_eigenvalues",
    "series_solution",
    "richardson",
    "observed_order",
    "convergence_orders",
    "self_convergence",
]

try:  # pragma: no cover - exercised only when scipy is present
    from scipy.special import j0 as _sp_j0
    from scipy.special import j1 as _sp_j1

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Bessel functions
# ---------------------------------------------------------------------------
def _j0_series(x: float) -> float:
    """Maclaurin series; accurate for |x| below about 8."""
    term = 1.0
    total = 1.0
    y = -0.25 * x * x
    for k in range(1, 40):
        term *= y / (k * k)
        total += term
        if abs(term) < 1e-18 * abs(total):
            break
    return total


def _j1_series(x: float) -> float:
    term = 0.5 * x
    total = term
    y = -0.25 * x * x
    for k in range(1, 40):
        term *= y / (k * (k + 1))
        total += term
        if abs(term) < 1e-18 * abs(total):
            break
    return total


def bessel_j0(x: float) -> float:
    """Bessel function of the first kind, order 0."""
    if _HAVE_SCIPY:
        return float(_sp_j0(x))
    x = float(x)
    if abs(x) < 12.0:
        return _j0_series(x)
    ax = abs(x)
    z = 12.0 / ax
    f = (
        0.79788456
        - 0.00000077 * z
        - 0.00552740 * z**2
        - 0.00009512 * z**3
        + 0.00137237 * z**4
        - 0.00072805 * z**5
        + 0.00014476 * z**6
    )
    th = ax - 0.78539816 - 0.04166397 * z - 0.00003954 * z**2 + 0.00262573 * z**3 \
        - 0.00054125 * z**4 - 0.00029333 * z**5 + 0.00013558 * z**6
    return f * math.cos(th) / math.sqrt(math.pi * ax)


def bessel_j1(x: float) -> float:
    """Bessel function of the first kind, order 1."""
    if _HAVE_SCIPY:
        return float(_sp_j1(x))
    x = float(x)
    if abs(x) < 12.0:
        return _j1_series(x)
    ax = abs(x)
    z = 12.0 / ax
    f = (
        0.79788456
        + 0.00000156 * z
        + 0.01659667 * z**2
        + 0.00017105 * z**3
        - 0.00249511 * z**4
        + 0.00113653 * z**5
        - 0.00020033 * z**6
    )
    th = ax - 2.35619449 + 0.12499612 * z + 0.00005650 * z**2 - 0.00637879 * z**3 \
        + 0.00074348 * z**4 + 0.00079824 * z**5 - 0.00029166 * z**6
    val = f * math.cos(th) / math.sqrt(math.pi * ax)
    return val if x > 0.0 else -val


# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------
def _bisect(f: Callable[[float], float], a: float, b: float, tol: float = 1e-14) -> float:
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5 * (a + b)


def robin_eigenvalues(geom: str, biot: float, n_modes: int) -> List[float]:
    """First ``n_modes`` positive roots of the Robin eigenvalue equation."""
    if biot < 0.0:
        raise ValueError("Biot number must be non-negative")

    if geom == "slab":
        fn = lambda m: m * math.sin(m) - biot * math.cos(m)
    elif geom == "cylinder":
        fn = lambda m: m * bessel_j1(m) - biot * bessel_j0(m)
    elif geom == "sphere":
        fn = lambda m: math.sin(m) * (1.0 - biot) - m * math.cos(m)
    else:
        raise ValueError("geom must be 'slab', 'cylinder' or 'sphere'")

    roots: List[float] = []
    x = 1e-6
    step = 0.05
    prev_x = x
    prev_f = fn(x)
    limit = 4.0 * n_modes + 40.0
    while x < limit and len(roots) < n_modes:
        x = prev_x + step
        cur = fn(x)
        if prev_f * cur < 0.0:
            root = _bisect(fn, prev_x, x)
            if root > 1e-8:
                roots.append(root)
        prev_x, prev_f = x, cur
        if x > 60.0:
            step = 0.5
    if len(roots) < n_modes:
        raise RuntimeError("could not find %d eigenvalues below mu=%.1f" % (n_modes, limit))
    return roots


# ---------------------------------------------------------------------------
# Series solution
# ---------------------------------------------------------------------------
def series_solution(
    geom: str,
    R: float,
    alpha: float,
    robin: float,
    u_init: float,
    u_env: float,
    n_modes: int = 400,
):
    """Closed-form solution of the constant-coefficient sibling problem.

    Returns a callable ``u(r, t)``. The initial condition is uniform, the
    ambient value is constant, and the boundary condition is Robin.

    ``u_init - u_env`` is signed. A cold body in a warm environment gives a
    negative offset, which flips every coefficient; using ``abs()`` here
    produces a reference that is smooth, plausible and wrong.

    The number of modes matters more than it looks. With a step initial
    condition and a large Biot number the coefficients decay slowly and
    alternate, so a few tens of modes leave an O(1) truncation error near
    ``t = 0``. The default of 400 keeps the truncation error well below the
    discretisation error being measured over the usual inspection window.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if R <= 0.0:
        raise ValueError("R must be positive")

    biot = robin * R / alpha
    mus = robin_eigenvalues(geom, biot, n_modes)
    theta0 = float(u_init) - float(u_env)

    amps: List[float] = []
    shapes_mean: List[float] = []

    for mu in mus:
        if geom == "slab":
            s, c = math.sin(mu), math.cos(mu)
            amp = 2.0 * theta0 * s / (mu + s * c)
            shapes_mean.append(s / mu)
        elif geom == "cylinder":
            j0, j1 = bessel_j0(mu), bessel_j1(mu)
            amp = 2.0 * theta0 * j1 / (mu * (j0 * j0 + j1 * j1))
            # volume-weighted mean of J0(mu*xi) on [0,1] is 2*J1(mu)/mu
            shapes_mean.append(2.0 * j1 / mu)
        else:
            s, c = math.sin(mu), math.cos(mu)
            den = mu * mu - mu * s * c
            amp = 2.0 * theta0 * (s - mu * c) / den
            shapes_mean.append(3.0 * (s - mu * c) / (mu * mu))
        amps.append(amp)

    modes = list(zip(mus, amps))

    def u(r: float, t: float) -> float:
        xi = r / R
        if xi < 0.0 or xi > 1.0:
            raise ValueError("r must lie in [0, R]")
        acc = 0.0
        for mu, amp in modes:
            if geom == "slab":
                shape = math.cos(mu * xi)
            elif geom == "cylinder":
                shape = bessel_j0(mu * xi)
            else:
                # eigenfunction is sin(mu*xi)/xi, whose limit at the origin is mu
                shape = mu if xi == 0.0 else math.sin(mu * xi) / xi
            e = -mu * mu * alpha * t / (R * R)
            if e < -700.0:
                break
            acc += amp * shape * math.exp(e)
        return float(u_env) + acc

    def u_mean(t: float) -> float:
        acc = 0.0
        for (mu, amp), sm in zip(modes, shapes_mean):
            e = -mu * mu * alpha * t / (R * R)
            if e < -700.0:
                break
            acc += amp * sm * math.exp(e)
        return float(u_env) + acc

    u.mean = u_mean  # type: ignore[attr-defined]
    u.eigenvalues = mus  # type: ignore[attr-defined]
    u.biot = biot  # type: ignore[attr-defined]
    return u


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def richardson(f_medium: float, f_fine: float, ratio: float = 2.0, order: float = 2.0) -> float:
    """Extrapolate a value to zero step size from two refinement levels."""
    return f_fine + (f_fine - f_medium) / (ratio**order - 1.0)


def observed_order(
    e_coarse: float, e_medium: float, e_fine: float, ratio: float = 2.0
) -> float:
    """Observed order of accuracy from three nested refinements.

    Returns ``0.0`` when the error is not shrinking at all and ``nan`` when an
    error is not positive, which makes the logarithm undefined. A flat error is
    a real reading: it usually means the reference value is wrong rather than
    the scheme, but the estimator should not hide it.
    """
    if e_coarse <= 0.0 or e_medium <= 0.0 or e_fine <= 0.0:
        return float("nan")
    r1 = e_coarse / e_medium
    if r1 <= 0.0:
        return float("nan")
    return math.log(r1) / math.log(ratio)


def convergence_orders(
    values: Sequence[float], reference: float, ratio: float = 2.0
) -> dict:
    """Errors and pairwise orders for a refinement sequence (coarse to fine).

    ``values`` must run from the coarsest to the finest level.
    """
    errors = [abs(v - reference) for v in values]
    orders = []
    for i in range(len(errors) - 2):
        orders.append(observed_order(errors[i], errors[i + 1], errors[i + 2], ratio))
    return {"errors": errors, "orders": orders, "reference": reference}


def self_convergence(values: Sequence[float], ratio: float = 2.0) -> dict:
    """Orders from consecutive differences, with no reference value needed.

    Refining one axis while holding the other fixed leaves the error on the
    fixed axis untouched, so the differences between successive solutions
    isolate the axis being refined. This is the practical way to measure the
    temporal order once the spatial error dominates a comparison against a
    closed form.

    ``values`` runs from coarsest to finest. An order near zero means the
    differences are dominated by something other than the refined axis.
    """
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    orders = []
    for i in range(len(diffs) - 1):
        if diffs[i] <= 0.0 or diffs[i + 1] <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(math.log(diffs[i] / diffs[i + 1]) / math.log(ratio))
    return {"differences": diffs, "orders": orders}
