"""recovery_explore — offline exploration of failure-state recovery.

Purely additive package: it imports this repo's
``cf_bench`` / ``rpc`` modules and copy-and-adapts the vendored RPent sources
under ``vendor/rpent/``, and it modifies nothing outside this directory.

Intentionally empty of logic so that importing the package never drags in
robosuite/robocasa; every env-dependent import lives inside a function.
"""

from __future__ import annotations
