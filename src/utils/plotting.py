"""Shared matplotlib configuration for headless PNG rendering across offline scripts and tests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt"]

