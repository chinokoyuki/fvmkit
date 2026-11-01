import random

import pytest

from fvmkit.geometry import Grid
from fvmkit.kernel import (
    Field,
    face_coef,
    solve_coupled,
    solve_field,
    solve_moving,
    thomas,
)
from fvmkit.verify import convergence_orders, self_convergence, series_solution


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def _tridiagonal_product(lo, dg, up, x):
    n = len(dg)
    out = [dg[i] * x[i] for i in range(n)]
    for i in range(1, n):
        out[i] += lo[i] * x[i - 1]
    for i in range(n - 1):
        out[i] += up[i] * x[i + 1]
    return out


def test_thomas_solves_a_diagonally_dominant_system():
    rng = random.Random(7)
    n = 41
    lo = [0.0] + [-rng.uniform(0.1, 1.0) for _ in range(n - 1)]
    up = [-rng.uniform(0.1, 1.0) for _ in range(n - 1)] + [0.0]
    dg = [abs(lo[i]) + abs(up[i]) + rng.uniform(0.5, 2.0) for i in range(n)]
    x_true = [rng.uniform(-2.0, 2.0) for _ in range(n)]
    rhs = _tridiagonal_product(lo, dg, up, x_true)

    x = thomas(lo, dg, up, rhs)
    for a, b in zip(x, x_true):
        assert a == pytest.approx(b, abs=1e-10)


def test_thomas_does_not_mutate_inputs():
    lo = [0.0, -1.0, -1.0]
    dg = [2.0, 2.0, 2.0]
    up = [-1.0, -1.0, 0.0]
    rhs = [1.0, 0.0, 1.0]
    snapshot = (list(lo), list(dg), list(up), list(rhs))
    thomas(lo, dg, up, rhs)
    assert (lo, dg, up, rhs) == snapshot


def test_face_coef_harmonic_vs_arithmetic():
    nodes = [1.0, 1000.0]
    assert face_coef(nodes)[0] == pytest.approx(2 * 1.0 * 1000.0 / 1001.0)
    assert face_coef(nodes, "arithmetic")[0] == pytest.approx(500.5)


def test_face_coef_stays_positive():
    assert face_coef([0.0, 5.0])[0] == 0.0
    assert face_coef([0.0, 0.0])[0] == 0.0


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geom", ["slab", "cylinder", "sphere"])
def test_conservation_residual_is_machine_precision(geom):
    g = Grid(0.02, 80, geom)
    hist, stats = solve_field(
        g,
        [1.0] * (g.N + 1),
        1.0,
        300,
        lambda u: [1e-9] * (g.N + 1),
        5e-7,
        lambda t: 0.0,
    )
    assert stats["fields"]["u"]["residual"] < 1e-11
    assert stats["unconverged_steps"] == 0


def test_accounting_gap_is_large_when_capacity_drifts():
    """A variable capacity makes the engineering accounting miss the identity."""
    g = Grid(0.02, 60, "cylinder")
    RHO_CP = lambda c: 820.0 * (1450.0 + 2736.0 * c / (c + 1.0))

    def props(state):
        return [RHO_CP(c) for c in state["c"]]

    fields = [
        Field(name="c", init=2.5, conductivity=lambda s: [1e-8] * (g.N + 1),
              robin=8e-7, env=lambda t: 0.1),
        Field(name="T", init=300.0, conductivity=lambda s: [0.36] * (g.N + 1),
              robin=25.0, env=lambda t: 350.0, capacity=lambda s: [g.v[i] * props(s)[i] for i in range(g.N + 1)]),
    ]
    hist, stats = solve_coupled(g, fields, 2.0, 300, theta=0.5)
    assert stats["fields"]["T"]["residual"] < 1e-10
    assert abs(stats["fields"]["T"]["gap_engineering"]) > 1e-3


# ---------------------------------------------------------------------------
# Agreement with the closed-form series and second-order convergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geom", ["slab", "cylinder", "sphere"])
def test_matches_series_solution_and_converges_at_second_order(geom):
    R, alpha, robin, T = 0.02, 1e-9, 5e-7, 900.0
    ref = series_solution(geom, R, alpha, robin, 1.0, 0.0, n_modes=3000)
    target = ref.mean(T)

    values = []
    for N in (40, 80, 160):
        g = Grid(R, N, geom)
        hist, _ = solve_field(
            g,
            [1.0] * (g.N + 1),
            1.0,
            int(T),
            lambda u: [alpha] * (g.N + 1),
            robin,
            lambda t: 0.0,
        )
        values.append(g.mean(hist[-1]))

    res = convergence_orders(values, target, ratio=2.0)
    assert res["errors"][-1] < 5e-5
    assert 1.85 < res["orders"][-1] < 2.15


def test_time_step_refinement_is_second_order():
    """Refine time only and use consecutive differences, so the fixed spatial
    error cancels out of the order estimate."""
    R, alpha, robin, T = 0.02, 1e-9, 5e-7, 400.0
    g = Grid(R, 80, "cylinder")

    values = []
    for nsteps in (25, 50, 100, 200):
        hist, _ = solve_field(
            g,
            [1.0] * (g.N + 1),
            T / nsteps,
            nsteps,
            lambda u: [alpha] * (g.N + 1),
            robin,
            lambda t: 0.0,
        )
        values.append(g.mean(hist[-1]))

    res = self_convergence(values, ratio=2.0)
    for order in res["orders"]:
        assert 1.9 < order < 2.1


def test_uniform_state_is_a_steady_state():
    g = Grid(0.02, 30, "sphere")
    hist, _ = solve_field(
        g, 4.2, 1.0, 50, lambda u: [1e-9] * (g.N + 1), 1e-3, lambda t: 4.2
    )
    for value in hist[-1]:
        assert value == pytest.approx(4.2, abs=1e-12)


# ---------------------------------------------------------------------------
# Coupling and material coordinates
# ---------------------------------------------------------------------------
def test_coupled_solver_converges_and_respects_both_fields():
    g = Grid(0.02, 40, "cylinder")
    fields = [
        Field("c", 1.0, lambda s: [1e-9] * (g.N + 1), 5e-7, lambda t: 0.0),
        Field("T", 300.0, lambda s: [1e-7] * (g.N + 1), 5e-6, lambda t: 350.0),
    ]
    hist, stats = solve_coupled(g, fields, 1.0, 200)
    assert stats["sweeps_max"] <= 8
    assert hist["c"][-1][0] < 1.0
    assert hist["T"][-1][-1] > 300.0


def test_moving_with_constant_radius_matches_fixed_grid():
    R, alpha, robin = 0.02, 1e-9, 5e-7
    N, dt, nsteps = 40, 2.0, 100

    fixed = Grid(R, N, "cylinder")
    hf, _ = solve_field(
        fixed, [1.0] * (N + 1), dt, nsteps, lambda u: [alpha] * (N + 1), robin, lambda t: 0.0
    )

    material = Grid(1.0, N, "cylinder")
    field = Field("u", 1.0, lambda s: [alpha] * (N + 1), robin, lambda t: 0.0)
    hm, _ = solve_moving(material, [field], dt, nsteps, lambda t: R)

    for a, b in zip(hf[-1], hm["u"][-1]):
        assert a == pytest.approx(b, rel=1e-10)


def test_moving_rejects_a_physical_grid():
    with pytest.raises(ValueError, match="material coordinates"):
        solve_moving(Grid(0.02, 10, "cylinder"), [], 1.0, 1, lambda t: 0.02)


def test_record_and_probe():
    g = Grid(0.02, 20, "cylinder")
    seen = []
    hist, stats = solve_field(
        g,
        [1.0] * (g.N + 1),
        1.0,
        10,
        lambda u: [1e-9] * (g.N + 1),
        5e-7,
        lambda t: 0.0,
        record=[0, 20],
        probe=lambda n, state: seen.append(n),
    )
    assert stats["record"] == [0, 20]
    assert len(hist[0]) == 2
    assert seen == list(range(1, 11))


def test_field_rejects_mismatched_initial_length():
    g = Grid(0.02, 10, "cylinder")
    with pytest.raises(ValueError, match="expected"):
        solve_field(g, [1.0, 1.0], 1.0, 5, lambda u: [1e-7] * (g.N + 1), 1e-3, lambda t: 0.0)
