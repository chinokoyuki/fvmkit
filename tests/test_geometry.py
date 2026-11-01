import pytest

from fvmkit.geometry import Grid


@pytest.mark.parametrize("geom,p", [("slab", 0), ("cylinder", 1), ("sphere", 2)])
def test_axis_coefficient_matches_lhopital_limit(geom, p):
    """The half cell at r=0 must reproduce 2(p+1)/dr^2 without a branch."""
    for N in (10, 37, 160):
        g = Grid(0.02, N, geom)
        expected = 2.0 * (p + 1) / g.dr**2
        assert g.outer_coefficient() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("geom,p", [("slab", 0), ("cylinder", 1), ("sphere", 2)])
def test_total_volume(geom, p):
    g = Grid(0.02, 64, geom)
    assert g.vol == pytest.approx(0.02 ** (p + 1) / (p + 1), rel=1e-12)


def test_mean_of_constant_is_that_constant():
    g = Grid(1.0, 40, "sphere")
    assert g.mean([3.5] * (g.N + 1)) == pytest.approx(3.5, rel=1e-14)


def test_mean_matches_quadrature_of_linear_field():
    """On a cylinder the volume average of r is 2R/3."""
    g = Grid(1.0, 400, "cylinder")
    assert g.mean(g.r) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_output_indices_accepts_nodal_radii():
    g = Grid(0.02, 200, "cylinder")
    idx = g.output_indices([0.0, 0.005, 0.01, 0.02])
    assert idx == [0, 50, 100, 200]


def test_output_indices_rejects_off_grid_radius():
    g = Grid(0.02, 200, "cylinder")
    with pytest.raises(ValueError, match="not on the grid"):
        g.output_indices([0.0, 0.00335])


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Grid(0.02, 10, "torus")
    with pytest.raises(ValueError):
        Grid(-1.0, 10)
    with pytest.raises(ValueError):
        Grid(1.0, 1)
