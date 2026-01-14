# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat, PixelColorEncoding

__all__ = ['opencv_to_bgr888']

_SUPPORTED_YUV_FORMATS = {'YUYV', 'UYVY', 'YVYU', 'NV12', 'NV21'}
_SUPPORTED_RGB_FORMATS = {
    'XRGB8888', 'BGRX8888', 'ARGB8888', 'BGRA8888',
    'XBGR8888', 'RGBX8888', 'ABGR8888', 'RGBA8888',
    'RGB888', 'BGR888',
}


def _can_use_opencv_yuv(fmt: PixelFormat, options: dict | None) -> bool:
    if fmt.name not in _SUPPORTED_YUV_FORMATS:
        return False

    # OpenCV only supports limited range bt601 YUV conversion
    if options:
        color_range = options.get('range', 'limited')
        if color_range != 'limited':
            return False
        encoding = options.get('encoding', 'bt601')
        if encoding != 'bt601':
            return False

    return True


def _can_use_opencv_raw(fmt: PixelFormat, options: dict | None) -> bool:
    # Only unpacked formats are supported (not packed 10P/12P)
    if fmt.packed:
        return False

    # Only use OpenCV if demosaic_method is 'opencv' or not specified
    if options:
        demosaic = options.get('demosaic_method')
        if demosaic is not None and demosaic != 'opencv':
            return False

    return True


def _can_use_opencv_rgb(fmt: PixelFormat) -> bool:
    """Check if OpenCV can handle this RGB format."""
    return fmt.name in _SUPPORTED_RGB_FORMATS


def opencv_to_bgr888(fmt: PixelFormat, width: int, height: int,
                     arr: npt.NDArray[np.uint8],
                     options: dict | None) -> npt.NDArray[np.uint8] | None:
    if fmt.color == PixelColorEncoding.YUV:
        if not _can_use_opencv_yuv(fmt, options):
            return None
    elif fmt.color == PixelColorEncoding.RAW:
        if not _can_use_opencv_raw(fmt, options):
            return None
    elif fmt.color == PixelColorEncoding.RGB:
        if not _can_use_opencv_rgb(fmt):
            return None
    else:
        return None

    # Import and call implementation only if format is supported
    from .opencv_impl import opencv_convert
    return opencv_convert(fmt, width, height, arr)
