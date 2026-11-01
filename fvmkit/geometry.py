"""Control-volume geometry for one-dimensional radial diffusion.

Slabs (``p = 0``), cylinders (``p = 1``) and spheres (``p = 2``) share a single
cell layout. Nodes sit on cell faces, so node 0 is the symmetry axis and node
``N`` is the outer surface::

    r_i        = i * dr                    i = 0 .. N
    r_{i+1/2}  = (i + 0.5) * dr            i = 0 .. N-1

Cell 0 is the half cell ``[0, dr/2]``; cell ``N`` is ``[R - dr/2, R]``. Volumes
and face areas are the shape integrals

    v_i = integral of r**p dr over cell i
    a_i = r_{i+1/2} ** p

The common ``2*pi`` (cylinder) or ``4*pi`` (sphere) factor appears in every
numerator and denominator of the discrete equations and is dropped. Dropping it
keeps the assembled coefficients O(1) and avoids needless round-off.

Because the first cell is half width, the ratio ``a_0 / (dr * v_0)`` evaluates to
``2*(p+1)/dr**2`` by construction. That is the L'Hopital limit of the cylindrical
or spherical Laplacian at the axis (2 for a slab, 4 for a cylinder, 6 for a
sphere), obtained without an ``if r == 0`` branch.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["Grid", "GEOMETRIES"]

GEOMETRIES = {"slab": 0, "cylinder": 1, "sphere": 2}


class Grid:
    """Radial control volumes on ``[0, R]``.

    Parameters
    ----------
    R:
        Outer radius. For an already normalised domain (material coordinates)
        pass ``1.0``.
    N:
        Number of cells. The grid then has ``N + 1`` nodes.
    geom:
        ``"slab"``, ``"cylinder"`` or ``"sphere"``.

    Attributes
    ----------
    r, rf:
        Node coordinates and interior face coordinates.
    v:
        Cell volumes (length ``N+1``).
    a:
        Interior face areas (length ``N``).
    a_R:
        Outer boundary area.
    vol:
        Total volume.
    """

    __slots__ = ("R", "N", "p", "geom", "dr", "r", "rf", "v", "a", "a_R", "vol")

    def __init__(self, R: float, N: int, geom: str = "cylinder") -> None:
        if geom not in GEOMETRIES:
            raise ValueError(
                "geom must be one of %s, got %r" % (sorted(GEOMETRIES), geom)
            )
        if R <= 0.0:
            raise ValueError("R must be positive")
        if N < 2:
            raise ValueError("N must be at least 2")

        self.R = float(R)
        self.N = int(N)
        self.geom = geom
        self.p = GEOMETRIES[geom]
        self.dr = self.R / self.N

        p, dr, N_, R_ = self.p, self.dr, self.N, self.R
        self.r = [i * dr for i in range(N_ + 1)]
        self.rf = [(i + 0.5) * dr for i in range(N_)]

        # v_i = (r_{i+1/2}^{p+1} - r_{i-1/2}^{p+1}) / (p+1), with r_{-1/2} = 0.
        inv = 1.0 / (p + 1)
        v = [0.0] * (N_ + 1)
        v[0] = (self.rf[0] ** (p + 1)) * inv
        for i in range(1, N_):
            v[i] = (self.rf[i] ** (p + 1) - self.rf[i - 1] ** (p + 1)) * inv
        v[N_] = (R_ ** (p + 1) - self.rf[N_ - 1] ** (p + 1)) * inv
        self.v = v

        self.a = [rf ** p for rf in self.rf]
        self.a_R = R_ ** p
        self.vol = sum(v)

    # -- helpers ---------------------------------------------------------
    def mean(self, u: Sequence[float]) -> float:
        """Volume average of a nodal field."""
        return sum(u[i] * self.v[i] for i in range(self.N + 1)) / self.vol

    def outer_coefficient(self) -> float:
        """``a_0 / (dr * v_0)``, the discrete axis coefficient."""
        return self.a[0] / (self.dr * self.v[0])

    def output_indices(self, radii: Sequence[float], rel_tol: float = 1e-9):
        """Node indices matching the requested radii.

        Every radius that is meant to be reported should land on a node. Doing
        so removes interpolation error from the exported values entirely.
        Raises ``ValueError`` listing the radii that do not sit on the grid.
        """
        dr = self.dr
        idx, missing = [], []
        for r in radii:
            if r < 0.0 or r > self.R * (1.0 + rel_tol):
                missing.append(r)
                continue
            k = int(round(r / dr))
            if abs(k * dr - r) > rel_tol * max(1.0, self.R):
                missing.append(r)
            else:
                idx.append(min(k, self.N))
        if missing:
            raise ValueError(
                "these radii are not on the grid (dr=%.6g): %s"
                % (dr, ", ".join("%.6g" % m for m in missing))
            )
        return idx

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Grid(R=%g, N=%d, geom=%r, dr=%.4g)" % (
            self.R,
            self.N,
            self.geom,
            self.dr,
        )
