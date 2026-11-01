"""Tabulated boundary data: shape-preserving interpolation, smooth continuation,
and continuous threshold-crossing times.

Three small pieces that show up together whenever a simulation is driven by a
sampled time series:

* :func:`pchip` - monotone cubic Hermite interpolation. No overshoot, so a
  non-physical hump cannot appear between two samples.
* :func:`blend_to_hold` - continues a tabulated curve past its last sample with
  a C1 transition into a constant. A hard cut at the last sample injects a
  step into the boundary condition.
* :func:`crossing_time` - resolves the instant a sampled signal passes a level,
  to well below the sampling interval.
"""

from __future__ import annotations

import sys
from typing import Callable, Sequence

__all__ = ["pchip", "smoothstep", "blend_to_hold", "crossing_time"]

_WARNED = set()


def _warn_once(xs: Sequence[float], x: float, side: str) -> None:
    key = (xs[0], xs[-1], side)
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(
        "[pchip] abscissa %.6g lies outside [%.6g, %.6g]; "
        "holding the end value instead of extrapolating"
        % (x, xs[0], xs[-1]),
        file=sys.stderr,
    )


def pchip(xs: Sequence[float], ys: Sequence[float]) -> Callable[[float], float]:
    """Shape-preserving cubic Hermite interpolant (Fritsch-Carlson).

    Monotone data produce a monotone interpolant: no overshoot at sharp steps.
    Outside ``[xs[0], xs[-1]]`` the end value is held flat and a warning is
    emitted once per side, because a silent flat hold is easy to mistake for a
    physical plateau.

    Returns a callable ``f(x)``.
    """
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have the same length")
    if n < 2:
        raise ValueError("at least two samples are required")

    span = (xs[-1] - xs[0]) if xs[-1] > xs[0] else 1.0
    tol = 1e-9 * span

    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    if any(d <= 0.0 for d in h):
        raise ValueError("xs must be strictly increasing")
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]

    m = [0.0] * n
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        if d0 == 0.0:
            return 0.0
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if s * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(s) > 3.0 * abs(d0):
            return 3.0 * d0
        return s

    if n == 2:
        m[0] = m[1] = d[0]
    else:
        m[0] = end_slope(h[0], h[1], d[0], d[1])
        m[-1] = end_slope(h[-1], h[-2], d[-1], d[-2])

    def f(x: float) -> float:
        if x <= xs[0]:
            if x < xs[0] - tol:
                _warn_once(xs, x, "lo")
            return ys[0]
        if x >= xs[-1]:
            if x > xs[-1] + tol:
                _warn_once(xs, x, "hi")
            return ys[-1]

        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        i = lo
        t = (x - xs[i]) / h[i]
        t2 = t * t
        return (
            (1.0 + 2.0 * t) * (1.0 - t) ** 2 * ys[i]
            + t * (1.0 - t) ** 2 * h[i] * m[i]
            + t2 * (3.0 - 2.0 * t) * ys[i + 1]
            + t2 * (t - 1.0) * h[i] * m[i + 1]
        )

    return f


def smoothstep(s: float) -> float:
    """``3s^2 - 2s^3``, clamped to ``[0, 1]``. Zero slope at both ends."""
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    return s * s * (3.0 - 2.0 * s)


def blend_to_hold(
    f: Callable[[float], float],
    x_stop: float,
    hold: float,
    taper: float,
) -> Callable[[float], float]:
    """Extend ``f`` beyond ``x_stop`` as the constant ``hold``, smoothly.

    The curve is left untouched up to ``x_stop - taper``, then blended with a
    smoothstep weight so that value and first derivative are continuous at both
    joins. ``taper <= 0`` falls back to a hard switch, which is available but
    rarely what you want.

    Returns a callable ``g(x)``.
    """
    if taper <= 0.0:

        def hard(x: float) -> float:
            return f(x) if x <= x_stop else hold

        return hard

    x_start = x_stop - taper

    def g(x: float) -> float:
        if x <= x_start:
            return f(x)
        if x >= x_stop:
            return hold
        w = smoothstep((x - x_start) / taper)
        return (1.0 - w) * f(x) + w * hold

    return g


def crossing_time(
    ts: Sequence[float],
    ys: Sequence[float],
    level: float,
    window: int = 6,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple:
    """First time a sampled, decreasing signal passes ``level``.

    Returns ``(t_star, info)``; ``t_star`` is ``None`` when the signal never
    crosses. The bracket is refined with a shape-preserving interpolant plus
    bisection, so the answer is not limited by the sampling interval.

    ``info`` reports the raw bracket, the residual of the interpolant at the
    root, the gap against a straight-line estimate over the bracket, and whether
    the prefix was monotone. A large gap is a hint that the signal is not
    resolved well enough for the linear reading to be trusted.
    """
    ts = list(ts)
    ys = list(ys)
    n = len(ys)
    if n != len(ts):
        raise ValueError("ts and ys must have the same length")

    k = None
    for i in range(1, n):
        if ys[i] < level <= ys[i - 1]:
            k = i
            break
    if k is None:
        return None, {"crossed": False}

    monotone = all(ys[i + 1] <= ys[i] + 1e-12 for i in range(min(k + 1, n - 1)))

    lo = max(0, k - window)
    hi = min(n - 1, k + window)
    xs = ts[lo : hi + 1]
    vs = ys[lo : hi + 1]
    f = pchip(xs, vs)

    a, b = ts[k - 1], ts[k]
    fa = ys[k - 1] - level
    fb = ys[k] - level
    if fa < 0.0:
        fa = f(a) - level

    for _ in range(max_iter):
        if b - a < tol:
            break
        m = 0.5 * (a + b)
        fm = f(m) - level
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    t_star = 0.5 * (a + b)
    slope = (ys[k - 1] - level) / (ys[k - 1] - ys[k]) if ys[k - 1] != ys[k] else 0.0
    linear = ts[k - 1] + (ts[k] - ts[k - 1]) * slope

    return t_star, {
        "crossed": True,
        "monotone_prefix": monotone,
        "index": k,
        "t_sampled": ts[k],
        "bracket": (ts[k - 1], ts[k]),
        "residual": f(t_star) - level,
        "linear_gap": t_star - linear,
    }
