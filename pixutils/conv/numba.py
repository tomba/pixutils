# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat, PixelColorEncoding

__all__ = ['numba_to_bgr888']

_SUPPORTED_YUV_FORMATS = {'YUYV', 'UYVY', 'NV12'}


def _can_use_numba_yuv(fmt: PixelFormat) -> bool:
    return fmt.name in _SUPPORTED_YUV_FORMATS


def _can_use_numba_raw(fmt: PixelFormat, options: dict | None) -> bool:
    # Numba supports all raw formats for unpacking and 3x3/bilinear demosaic
    # Check demosaic method - numba handles '3x3', 'bilinear', 'mosaic'
    if options:
        method = options.get('demosaic_method')
        if method is not None and method not in ('3x3', 'bilinear', 'mosaic'):
            return False
    return True


def numba_to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int,
    arr: npt.NDArray[np.uint8],
    options: dict | None,
) -> npt.NDArray[np.uint8] | None:
    """Try to convert using numba. Returns None if numba can't handle this format."""

    if fmt.color == PixelColorEncoding.YUV:
        if not _can_use_numba_yuv(fmt):
            return None
        from .yuv_nb import yuv_to_bgr888_nb

        return yuv_to_bgr888_nb(arr, width, height, fmt, options)

    if fmt.color == PixelColorEncoding.RAW:
        if not _can_use_numba_raw(fmt, options):
            return None
        from .raw_nb import raw_to_bgr888_nb

        return raw_to_bgr888_nb(arr, width, height, bytesperline, fmt, options)

    # RGB has no numba implementation (numpy is fast enough)
    return None
