import math

import pytest

from fvmkit.audit import (
    accounting_gap,
    accounting_rows,
    fit_decay_models,
    solve_target,
    sweep,
)
from fvmkit.geometry import Grid
from fvmkit.kernel import solve_field


def test_sweep_reports_relative_change():
    rows = sweep(lambda x: 2.0 * x, base=1.0, points=[1.0, 2.0, 4.0], label="scale")
    assert [r.value for r in rows] == [2.0, 4.0, 8.0]
    assert rows[0].delta_rel == pytest.approx(0.0)
    assert rows[2].delta_rel == pytest.approx(3.0)
    assert all(r.label == "scale" for r in rows)


def test_solve_target_recovers_a_known_root():
    x, value, iters = solve_target(lambda x: x * x, 9.0, 0.0, 10.0)
    assert x == pytest.approx(3.0, rel=1e-6)
    assert value == pytest.approx(9.0, rel=1e-6)
    assert iters > 0


def test_solve_target_works_on_a_decreasing_function():
    x, value, _ = solve_target(lambda x: 1.0 / x, 0.25, 1.0, 10.0)
    assert x == pytest.approx(4.0, rel=1e-6)


def test_solve_target_rejects_an_unbracketed_target():
    with pytest.raises(ValueError, match="not bracketed"):
        solve_target(lambda x: x, 100.0, 0.0, 1.0)


def test_accounting_gap_matches_the_closed_form():
    w_new = [2.0, 3.0, 4.0]
    w_old = [1.0, 1.5, 2.0]
    u_old = [300.0, 305.0, 310.0]
    ref = 273.15
    expected = sum((w_new[i] - w_old[i]) * (ref - u_old[i]) for i in range(3))
    assert accounting_gap(w_new, w_old, u_old, ref) == pytest.approx(expected)


def test_accounting_gap_vanishes_for_constant_capacity():
    assert accounting_gap([2.0] * 4, [2.0] * 4, [1.0, 2.0, 3.0, 4.0], 0.0) == 0.0


def test_accounting_rows_are_consistent_with_a_run():
    g = Grid(0.02, 30, "cylinder")
    _, stats = solve_field(g, 1.0, 1.0, 60, lambda u: [1e-9] * (g.N + 1), 5e-7, lambda t: 0.0)
    rows = accounting_rows(stats)
    assert len(rows) == 1
    assert rows[0]["field"] == "u"
    assert abs(rows[0]["storage_exact"] - rows[0]["flux"]) / abs(rows[0]["flux"]) < 1e-12


def test_fit_decay_models_recovers_page_parameters():
    k_true = 0.12
    ts = [0.5 + 0.5 * i for i in range(40)]
    mr = [math.exp(-k_true * t) for t in ts]  # Page with n = 1
    fits = fit_decay_models(ts, mr, models=("page", "henderson_pabis"))
    assert fits["page"]["rmse"] < 1e-6
    assert fits["page"]["params"][0] == pytest.approx(k_true, rel=0.05)
    assert fits["henderson_pabis"]["rmse"] < 1e-6


def test_fit_decay_models_reports_extrapolation_error():
    k_true = 0.1
    ts = [0.5 * i for i in range(1, 60)]
    mr = [math.exp(-k_true * t) for t in ts]
    fits = fit_decay_models(ts, mr, models=("page",), train_until=10.0)
    assert fits["page"]["rmse_extrapolated"] is not None
    assert fits["page"]["rmse_extrapolated"] < 1e-6


def test_fit_decay_models_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown model"):
        fit_decay_models([1.0, 2.0, 3.0], [1.0, 0.5, 0.2], models=("fractal",))
