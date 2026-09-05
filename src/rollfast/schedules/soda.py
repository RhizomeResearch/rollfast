"""Backward-compatible SODA imports.

SODA lives under :mod:`rollfast.optim.soda`; this module preserves the previous
``rollfast.schedules.soda`` import path.
"""

from rollfast.optim.soda import (
    soda,
    soda_adam,
    soda_kron,
    soda_muon,
    soda_prism,
    soda_rmnp,
)

__all__ = [
    "soda",
    "soda_adam",
    "soda_kron",
    "soda_muon",
    "soda_prism",
    "soda_rmnp",
]
