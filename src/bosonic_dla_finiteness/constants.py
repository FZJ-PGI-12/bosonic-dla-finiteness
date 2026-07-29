"""
Shared numerical tolerance.

ZERO_TOL is the threshold below which a real quantity is treated as exactly
zero.

Caveat
------
This is an absolute tolerance, so it presumes frequencies of order unity.
Rescaling the frequency units, such as expressing ω where typical values are ~1e-10,
makes the tolerance significant, and modes that are merely close in
frequency may then be classified as exactly degenerate.
"""

ZERO_TOL: float = 1e-12
