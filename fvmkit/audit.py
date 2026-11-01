"""Sensitivity sweeps, inverse calibration, accounting comparison, thin-layer fits.

The routines here are about turning a solver into a defensible number: how much
does the answer move when an input moves, what input would be needed to reach a
target, and does the reported residual actually measure what it claims to.

``accounting`` is the one worth reading first. Conservation audits routinely
report a large residual for a scheme that is in fact conservative, because the
accounting functional and the discretised functional disagree. The gap between
them has a closed form, so the two cases can be told apart instead of guessed
at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SweepRow",
    "sweep",
    "solve_target",
    "accounting_rows",
    "accounting_gap",
    "fit_decay_models",
]


# ---------------------------------------------------------------------------
# One-at-a-time sensitivity
# ---------------------------------------------------------------------------
@dataclass
class SweepRow:
    label: str
    param: float
    value: float
    delta_rel: float


def sweep(
    f: Callable[[float], float],
    base: float,
    points: Sequence[float],
    label: str = "",
) -> List[SweepRow]:
    """Evaluate ``f`` over ``points`` and report each result against ``f(base)``.

    The baseline is evaluated once and included only through ``delta_rel``; it is
    not added to the returned rows unless it appears in ``points``.
    """
    ref = f(base)
    rows = []
    for p in points:
        v = f(p)
        delta = (v - ref) / ref if ref != 0.0 else float("nan")
        rows.append(SweepRow(label=label, param=float(p), value=v, delta_rel=delta))
    return rows


def solve_target(
    f: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[float, float, int]:
    """Bisection for the parameter ``x`` with ``f(x) == target``.

    ``f`` must be monotone on ``[lo, hi]``; either direction is accepted. Returns
    ``(x, f(x), iterations)``. A bracket that does not straddle ``target`` raises
    ``ValueError`` rather than iterating blindly.
    """
    flo = f(lo) - target
    fhi = f(hi) - target
    if flo == 0.0:
        return lo, flo + target, 0
    if fhi == 0.0:
        return hi, fhi + target, 0
    if flo * fhi > 0.0:
        raise ValueError(
            "target %.6g is not bracketed by f(%.6g)=%.6g and f(%.6g)=%.6g"
            % (target, lo, flo + target, hi, fhi + target)
        )

    it = 0
    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - target
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if abs(hi - lo) < tol * max(1.0, abs(mid)):
            break
    x = 0.5 * (lo + hi)
    return x, f(x), it


# ---------------------------------------------------------------------------
# Conservation accounting
# ---------------------------------------------------------------------------
def accounting_rows(stats: Dict) -> List[Dict]:
    """Flatten ``stats["fields"]`` into one row per field.

    ``residual`` is the relative difference between the storage change computed
    with the new-time capacity and the integrated boundary flux. It should sit
    near machine precision for any theta in (0, 1].

    ``gap_conservative`` and ``gap_engineering`` are relative differences between
    that exact storage term and two other ways of bookkeeping the same step.
    They are not errors in the scheme; they measure how far the chosen reporting
    convention sits from the discrete identity.
    """
    rows = []
    for name, s in stats.get("fields", {}).items():
        rows.append(
            {
                "field": name,
                "flux": s["flux"],
                "storage_exact": s["storage_exact"],
                "residual": s["residual"],
                "gap_conservative": s["gap_conservative"],
                "gap_engineering": s["gap_engineering"],
            }
        )
    return rows


def accounting_gap(
    w_new: Sequence[float],
    w_old: Sequence[float],
    u_old: Sequence[float],
    reference: float,
) -> float:
    """Closed-form difference between exact and reference-based accounting.

    For one step, the exact storage term is ``sum W_new (u_new - u_old)`` while
    the reference-based one is ``sum [W_new (u_new - ref) - W_old (u_old - ref)]``.
    Subtracting gives::

        gap = sum (W_new - W_old) * (ref - u_old)

    When the capacity is constant the gap vanishes. When it is not, the gap is
    first order in the capacity change and does not shrink under refinement, so
    a large value here is a statement about the bookkeeping convention rather
    than about the discrete scheme.
    """
    if len(w_new) != len(w_old) or len(w_new) != len(u_old):
        raise ValueError("arrays must have the same length")
    return sum((w_new[i] - w_old[i]) * (reference - u_old[i]) for i in range(len(w_new)))


# ---------------------------------------------------------------------------
# Thin-layer drying models
# ---------------------------------------------------------------------------
def _nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    step: float = 0.2,
    tol: float = 1e-12,
    max_iter: int = 4000,
) -> Tuple[List[float], float]:
    """Plain Nelder-Mead. Sufficient for the low-dimensional fits below."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step if p[i] == 0.0 else step * abs(p[i])
        simplex.append(p)
    vals = [f(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]

        def along(t: float) -> List[float]:
            return [centroid[j] + t * (worst[j] - centroid[j]) for j in range(n)]

        xr = along(-1.0)
        fr = f(xr)
        if fr < vals[0]:
            xe = along(-2.0)
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = along(-0.5)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [0.5 * (simplex[i][j] + simplex[0][j]) for j in range(n)]
                    vals[i] = f(simplex[i])

    best = min(range(n + 1), key=lambda i: vals[i])
    return simplex[best], vals[best]


_MODELS = {
    # name: (n_params, function(t, *p))
    "page": (2, lambda t, k, n: math.exp(-k * t**n)),
    "henderson_pabis": (2, lambda t, a, k: a * math.exp(-k * t)),
    "midilli": (4, lambda t, a, k, n, b: a * math.exp(-k * t**n) + b * t),
}

_GUESS = {
    "page": (0.1, 1.0),
    "henderson_pabis": (1.0, 0.1),
    "midilli": (1.0, 0.1, 1.0, 0.0),
}


def fit_decay_models(
    ts: Sequence[float],
    mr: Sequence[float],
    models: Sequence[str] = ("page", "henderson_pabis", "midilli"),
    train_until: Optional[float] = None,
) -> Dict[str, Dict]:
    """Fit standard thin-layer moisture-ratio models to a decay curve.

    ``mr`` is the moisture ratio, already normalised to start near 1 and decay
    toward 0. ``ts`` is in the same units used by the model exponent.

    When ``train_until`` is given, each model is calibrated on ``ts <=
    train_until`` and its error is reported over the full range as well. That
    split is the informative one: all three models interpolate well, and they
    differ in how far they extrapolate.

    Returns ``{name: {"params": [...], "rmse": float, "rmse_extrapolated":
    float or None}}``.
    """
    ts = list(ts)
    mr = list(mr)
    if len(ts) != len(mr):
        raise ValueError("ts and mr must have the same length")

    if train_until is None:
        train = list(range(len(ts)))
    else:
        train = [i for i, t in enumerate(ts) if t <= train_until]
        if len(train) < 4:
            raise ValueError("not enough points before train_until=%g" % train_until)

    out: Dict[str, Dict] = {}
    for name in models:
        if name not in _MODELS:
            raise ValueError("unknown model %r; choose from %s" % (name, sorted(_MODELS)))
        n_par, fn = _MODELS[name]

        def loss(p, _fn=fn, _tr=train):
            s = 0.0
            for i in _tr:
                pred = _fn(ts[i], *p)
                s += (pred - mr[i]) ** 2
            return s / len(_tr)

        best, _ = _nelder_mead(loss, _GUESS[name])

        def rmse(indices, _fn=fn, _p=best):
            if not indices:
                return float("nan")
            s = 0.0
            for i in indices:
                s += (_fn(ts[i], *_p) - mr[i]) ** 2
            return math.sqrt(s / len(indices))

        full = list(range(len(ts)))
        out[name] = {
            "params": [float(x) for x in best],
            "rmse": rmse(full),
            "rmse_train": rmse(train),
            "rmse_extrapolated": None if train_until is None else rmse(full[len(train):]),
        }
    return out
