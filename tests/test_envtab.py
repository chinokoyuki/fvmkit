import math
import random

import pytest

from fvmkit.envtab import blend_to_hold, crossing_time, pchip, smoothstep


def test_pchip_reproduces_samples():
    xs = [0.0, 1.0, 2.5, 4.0, 7.0]
    ys = [0.0, 2.0, 1.0, 3.0, 3.5]
    f = pchip(xs, ys)
    for x, y in zip(xs, ys):
        assert f(x) == pytest.approx(y, abs=1e-12)


def test_pchip_is_monotone_on_monotone_data():
    xs = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    ys = [0.0, 0.9, 1.0, 1.02, 1.03, 1.031]
    f = pchip(xs, ys)
    prev = f(xs[0])
    x = xs[0]
    while x < xs[-1]:
        x += 0.01
        cur = f(x)
        assert cur >= prev - 1e-12
        prev = cur


def test_pchip_does_not_overshoot_a_step():
    xs = [0.0, 1.0, 1.0001]
    ys = [0.0, 0.0, 1.0]
    f = pchip(xs, ys)
    for i in range(1001):
        x = 0.9 + 0.2 * i / 1000.0
        assert -1e-12 <= f(x) <= 1.0 + 1e-12


def test_pchip_holds_end_values(capsys):
    f = pchip([0.0, 1.0], [2.0, 5.0])
    assert f(-3.0) == 2.0
    assert f(4.0) == 5.0
    assert "outside" in capsys.readouterr().err


def test_pchip_rejects_non_increasing_abscissae():
    with pytest.raises(ValueError):
        pchip([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])


def test_smoothstep_endpoints_and_clamping():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(-5.0) == 0.0
    assert smoothstep(5.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)


def test_blend_is_c1_at_both_joins():
    f = lambda t: 100.0 + 20.0 * t
    g = blend_to_hold(f, x_stop=10.0, hold=50.0, taper=4.0)

    # value continuity
    assert g(6.0) == pytest.approx(f(6.0))
    assert g(10.0) == pytest.approx(50.0)
    assert g(12.0) == pytest.approx(50.0)

    # first derivative continuity, by central differences
    h = 1e-5
    d_left = (g(6.0 + h) - g(6.0 - h)) / (2 * h)
    d_right = (g(10.0 + h) - g(10.0 - h)) / (2 * h)
    assert d_left == pytest.approx(20.0, rel=1e-4)
    assert abs(d_right) < 1e-3


def test_blend_without_taper_switches_hard():
    g = blend_to_hold(lambda t: 2.0 * t, x_stop=5.0, hold=0.0, taper=0.0)
    assert g(4.999) == pytest.approx(9.998)
    assert g(5.001) == 0.0


def test_crossing_time_beats_the_sampling_interval():
    # exponential decay sampled once per second; the exact crossing of 0.5 sits
    # at t = ln(2) / 0.7, between two samples. The cubic interpolant is good to
    # a few times 1e-4 over a one-second cell, so the answer is roughly three
    # orders of magnitude better than the sample spacing.
    k = 0.7
    ts = [i * 1.0 for i in range(20)]
    ys = [math.exp(-k * t) for t in ts]
    t_star, info = crossing_time(ts, ys, 0.5)
    exact = math.log(2.0) / k
    assert info["crossed"]
    assert t_star == pytest.approx(exact, abs=1e-3)
    assert abs(info["linear_gap"]) > 5.0 * abs(t_star - exact)


def test_crossing_time_is_accurate_with_dense_sampling():
    k = 0.7
    ts = [0.1 * i for i in range(100)]
    ys = [math.exp(-k * t) for t in ts]
    t_star, _ = crossing_time(ts, ys, 0.5)
    assert t_star == pytest.approx(math.log(2.0) / k, abs=1e-5)


def test_crossing_time_reports_no_crossing():
    t_star, info = crossing_time([0.0, 1.0, 2.0], [5.0, 4.0, 3.0], 0.5)
    assert t_star is None
    assert not info["crossed"]


def test_crossing_time_flags_non_monotone_prefix():
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 0.6, 0.8, 0.4, 0.2]
    t_star, info = crossing_time(ts, ys, 0.5)
    assert t_star is not None
    assert info["crossed"]
