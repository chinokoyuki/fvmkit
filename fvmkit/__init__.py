"""Conservative finite-volume kernels for radial diffusion, plus verification tools.

Submodules
----------
geometry
    Control-volume layout for slabs, cylinders and spheres.
kernel
    Theta-method / Picard solver for one or more coupled fields.
envtab
    Shape-preserving interpolation, smooth continuation, threshold crossing.
verify
    Analytic reference solutions, convergence order, conservation checks.
audit
    Parameter sweeps, inverse calibration, accounting-mode comparison.
docsync
    Numeric consistency checks between generated results and written text.
"""

from .geometry import Grid
from .kernel import Field, face_coef, solve_coupled, solve_field, solve_moving, thomas

__version__ = "0.1.0"

__all__ = [
    "Grid",
    "Field",
    "thomas",
    "face_coef",
    "solve_field",
    "solve_coupled",
    "solve_moving",
    "__version__",
]
