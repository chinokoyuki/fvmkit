import math

import pytest

from fvmkit.verify import (
    bessel_j0,
    bessel_j1,
    convergence_orders,
    observed_order,
    richardson,
    robin_eigenvalues,
    self_convergence,
    series_solution,
)


def test_bessel_values():
    assert bessel_j0(0.0) == pytest.approx(1.0, abs=1e-12)
    assert bessel_j0(2.404825557695773) == pytest.approx(0.0, abs=1e-6)
    assert bessel_j1(0.0) == pytest.approx(0.0, abs=1e-12)
    assert bessel_j1(2.0) == pytest.approx(0.5767248077568734, rel=1e-9)
    assert bessel_j0(-1.5) == pytest.approx(bessel_j0(1.5))
    assert bessel_j1(-1.5) == pytest.approx(-bessel_j1(1.5))


def test_eigenvalues_for_an_insulated_boundary():
    # Bi = 0: slab roots are n*pi, cylinder roots are zeros of J1
    slab = robin_eigenvalues("slab", 0.0, 3)
    assert slab[0] == pytest.approx(math.pi, rel=1e-6)
    cyl = robin_eigenvalues("cylinder", 0.0, 2)
    assert cyl[0] == pytest.approx(3.831705970, rel=1e-6)
    assert cyl[1] == pytest.approx(7.015586670, rel=1e-6)


def test_eigenvalues_approach_the_dirichlet_limit():
    # as Bi grows, the first root approaches the first zero of J0
    cyl = robin_eigenvalues("cylinder", 1e6, 1)
    assert cyl[0] == pytest.approx(2.4048255577, rel=1e-4)


def test_series_reproduces_the_initial_state_as_a_volume_average():
    for geom in ("slab", "cylinder", "sphere"):
        f = series_solution(geom, 0.02, 1e-9, 5e-7, 1.0, 0.0, n_modes=1500)
        assert f.mean(0.0) == pytest.approx(1.0, abs=1e-9)


def test_series_tends_to_the_ambient_value():
    f = series_solution("cylinder", 0.02, 1e-9, 5e-7, 1.0, 0.0, n_modes=400)
    assert f(0.01, 1e7) == pytest.approx(0.0, abs=1e-9)


def test_series_honours_a_signed_offset():
    """A cold start gives a negative offset, and the result must sit below ambient."""
    f = series_solution("slab", 0.02, 1e-9, 5e-7, 280.0, 350.0, n_modes=1000)
    assert f.mean(0.0) == pytest.approx(280.0, abs=1e-7)
    assert f(0.01, 600.0) < 350.0


def test_series_rejects_an_out_of_range_radius():
    f = series_solution("cylinder", 0.02, 1e-9, 5e-7, 1.0, 0.0, n_modes=50)
    with pytest.raises(ValueError):
        f(0.05, 10.0)


def test_richardson_and_observed_order_are_consistent():
    # synthetic second-order convergence toward 1.0
    exact = 1.0
    errors = [4.0e-4, 1.0e-4, 2.5e-5]
    values = [exact + e for e in errors]
    assert observed_order(errors[0], errors[1], errors[2]) == pytest.approx(2.0)
    extrapolated = richardson(values[1], values[2], ratio=2.0, order=2.0)
    assert extrapolated == pytest.approx(exact, abs=1e-12)


def test_observed_order_is_zero_when_errors_do_not_shrink():
    assert observed_order(1e-3, 1e-3, 1e-3) == pytest.approx(0.0)


def test_observed_order_is_nan_for_degenerate_errors():
    assert math.isnan(observed_order(0.0, 1e-4, 1e-5))


def test_convergence_orders_length():
    res = convergence_orders([1.1, 1.02, 1.005], reference=1.0, ratio=2.0)
    assert len(res["errors"]) == 3
    assert len(res["orders"]) == 1


def test_self_convergence_ignores_a_constant_bias():
    # a fixed offset does not change the differences
    values = [1.0 + 4e-4, 1.0 + 1e-4, 1.0 + 2.5e-5, 1.0 + 6.25e-6]
    res = self_convergence(values, ratio=2.0)
    for order in res["orders"]:
        assert order == pytest.approx(2.0, abs=1e-9)
