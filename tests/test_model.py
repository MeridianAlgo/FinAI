"""Deprecated test harness.

This module previously contained a verbose, print-driven test script that was
run as a standalone program. It interfered with pytest's collection and
output capture. Proper pytest tests live in `tests/test_model_pytest.py`.

If you need the old behavior, use `python scripts/legacy_test_model.py`.
"""

__all__ = []
