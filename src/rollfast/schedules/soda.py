"""Backward-compatible SODA imports.

SODA lives under :mod:`rollfast.optim.soda`; this module preserves the previous
``rollfast.schedules.soda`` import path.
"""

from rollfast.optim.soda import soda as soda
from rollfast.optim.soda import soda_adam as soda_adam
from rollfast.optim.soda import soda_kron as soda_kron
from rollfast.optim.soda import soda_muon as soda_muon
from rollfast.optim.soda import soda_prism as soda_prism
from rollfast.optim.soda import soda_rmnp as soda_rmnp

__all__ = [
    "soda",
    "soda_adam",
    "soda_kron",
    "soda_muon",
    "soda_prism",
    "soda_rmnp",
]
