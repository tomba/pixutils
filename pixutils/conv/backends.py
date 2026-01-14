# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

"""Backend discovery and selection for pixel format conversions."""

from __future__ import annotations

import importlib.util
import os

__all__ = ['get_backends']


def _get_available_backends() -> list[str]:
    env = os.environ.get('PIXUTILS_BACKENDS')
    if env:
        backends = [b.strip() for b in env.split(',')]
    else:
        backends = ['opencv', 'numba', 'numpy']

    available = []
    for backend in backends:
        if backend == 'numpy':
            available.append('numpy')
        elif backend == 'numba':
            if importlib.util.find_spec('numba'):
                available.append('numba')
        elif backend == 'opencv':
            if importlib.util.find_spec('cv2'):
                available.append('opencv')
        else:
            raise ValueError(f"Invalid backend '{backend}' in PIXUTILS_BACKENDS.")

    return available


_available_backends: list[str] = _get_available_backends()


def get_backends(requested: list[str] | None = None) -> list[str]:
    """Return list of backends to try, in priority order.

    Args:
        requested: Optional filter list from options['backends'].
                   If None, returns all available backends.

    Returns:
        Available backends filtered by request, in priority order.
    """
    if requested is None:
        return _available_backends.copy()
    return [b for b in requested if b in _available_backends]
