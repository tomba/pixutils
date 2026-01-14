# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from typing import Callable, cast

import cv2 # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat, PixelColorEncoding

__all__ = ['opencv_convert']

BAYER_PATTERN_MAP: dict[str, int] = {
    'RGGB': cv2.COLOR_BAYER_RG2BGR,
    'BGGR': cv2.COLOR_BAYER_BG2BGR,
    'GRBG': cv2.COLOR_BAYER_GR2BGR,
    'GBRG': cv2.COLOR_BAYER_GB2BGR,
}

# Tuple is (cv2 color code or None, reshape function)
RGB_FORMAT_MAP: dict[str, tuple[int | None, Callable]] = {
    # 32-bit BGRA formats
    'XRGB8888': (cv2.COLOR_BGRA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    'BGRX8888': (cv2.COLOR_BGRA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    'ARGB8888': (cv2.COLOR_BGRA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    'BGRA8888': (cv2.COLOR_BGRA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    # 32-bit RGBA formats
    'XBGR8888': (cv2.COLOR_RGBA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    'RGBX8888': (cv2.COLOR_RGBA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    'ABGR8888': (cv2.COLOR_RGBA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    'RGBA8888': (cv2.COLOR_RGBA2BGR, lambda b, w, h: b.reshape(h, w, 4)),
    # 24-bit formats
    'RGB888': (cv2.COLOR_RGB2BGR, lambda b, w, h: b.reshape(h, w, 3)),
    'BGR888': (None, lambda b, w, h: b.reshape(h, w, 3)),
}

# Tuple is (cv2 color code or None, reshape function)
YUV_FORMAT_MAP: dict[str, tuple[int, Callable]] = {
    'YUYV': (cv2.COLOR_YUV2BGR_YUY2, lambda b, w, h: b.reshape(h, w, 2)),
    'UYVY': (cv2.COLOR_YUV2BGR_UYVY, lambda b, w, h: b.reshape(h, w, 2)),
    'YVYU': (cv2.COLOR_YUV2BGR_YVYU, lambda b, w, h: b.reshape(h, w, 2)),
    'NV12': (cv2.COLOR_YUV2BGR_NV12, lambda b, w, h: b.reshape(h * 3 // 2, w)),
    'NV21': (cv2.COLOR_YUV2BGR_NV21, lambda b, w, h: b.reshape(h * 3 // 2, w)),
}


def _convert_yuv(fmt: PixelFormat, width: int, height: int,
                 arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    cv_code, reshape_func = YUV_FORMAT_MAP[fmt.name]
    reshaped = reshape_func(arr, width, height)
    return cv2.cvtColor(reshaped, cv_code)


def _convert_rgb(fmt: PixelFormat, width: int, height: int,
                 arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    cv_code, reshape_func = RGB_FORMAT_MAP[fmt.name]
    reshaped = reshape_func(arr, width, height)

    if cv_code is None:
        # Already BGR, just return a copy
        return reshaped.copy()

    return cv2.cvtColor(reshaped, cv_code)


def _convert_raw(fmt: PixelFormat, width: int, height: int,
                 arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8] | None:
    pattern = fmt.bayer_pattern
    assert pattern is not None
    cv_code = BAYER_PATTERN_MAP[pattern]

    name = fmt.name

    # Determine bit depth from format name
    if '8' in name:
        # 8-bit: reshape to (h, w) uint8
        bayer = arr.reshape(height, width)
        return cast(npt.NDArray[np.uint8], cv2.cvtColor(bayer, cv_code))
    elif '16' in name:
        # 16-bit: reshape to (h, w) uint16, convert, then scale to 8-bit
        bayer = arr.view(np.uint16).reshape(height, width)
        bgr16 = cast(npt.NDArray[np.uint16], cv2.cvtColor(bayer, cv_code))
        return (bgr16 >> 8).astype(np.uint8)
    elif '10' in name or '12' in name:
        # 10/12-bit unpacked (stored in 16-bit): shift up to use full 16-bit range
        bits = 10 if '10' in name else 12
        bayer = arr.view(np.uint16).reshape(height, width)
        bayer = bayer << (16 - bits)
        bgr16 = cast(npt.NDArray[np.uint16], cv2.cvtColor(bayer, cv_code))
        return (bgr16 >> 8).astype(np.uint8)
    else:
        # Unknown bit depth
        return None


def opencv_convert(fmt: PixelFormat, width: int, height: int,
                   arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8] | None:
    if fmt.color == PixelColorEncoding.YUV:
        return _convert_yuv(fmt, width, height, arr)

    if fmt.color == PixelColorEncoding.RAW:
        return _convert_raw(fmt, width, height, arr)

    if fmt.color == PixelColorEncoding.RGB:
        return _convert_rgb(fmt, width, height, arr)

    return None
