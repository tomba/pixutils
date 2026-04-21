# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from typing import cast

import cv2  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided

from pixutils.formats import PixelFormat, PixelColorEncoding

__all__ = ['opencv_convert']

BAYER_PATTERN_MAP: dict[str, int] = {
    'RGGB': cv2.COLOR_BAYER_RG2BGR,
    'BGGR': cv2.COLOR_BAYER_BG2BGR,
    'GRBG': cv2.COLOR_BAYER_GR2BGR,
    'GBRG': cv2.COLOR_BAYER_GB2BGR,
}

RGB_FORMAT_MAP: dict[str, int | None] = {
    # 32-bit BGRA formats
    'XRGB8888': cv2.COLOR_BGRA2BGR,
    'BGRX8888': cv2.COLOR_BGRA2BGR,
    'ARGB8888': cv2.COLOR_BGRA2BGR,
    'BGRA8888': cv2.COLOR_BGRA2BGR,
    # 32-bit RGBA formats
    'XBGR8888': cv2.COLOR_RGBA2BGR,
    'RGBX8888': cv2.COLOR_RGBA2BGR,
    'ABGR8888': cv2.COLOR_RGBA2BGR,
    'RGBA8888': cv2.COLOR_RGBA2BGR,
    # 24-bit formats
    'RGB888': cv2.COLOR_RGB2BGR,
    'BGR888': None,  # Already BGR
}

YUV_FORMAT_MAP: dict[str, int] = {
    'YUYV': cv2.COLOR_YUV2BGR_YUY2,
    'UYVY': cv2.COLOR_YUV2BGR_UYVY,
    'YVYU': cv2.COLOR_YUV2BGR_YVYU,
    'NV12': cv2.COLOR_YUV2BGR_NV12,
    'NV21': cv2.COLOR_YUV2BGR_NV21,
}


def _convert_yuv(
    fmt: PixelFormat, width: int, height: int, stride: int, arr: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    cv_code = YUV_FORMAT_MAP[fmt.name]

    if len(fmt.planes) == 1:
        # Packed formats (YUYV, UYVY, YVYU)
        plane = fmt.planes[0]
        bytes_per_pixel = plane.bytes_per_block // plane.pixels_per_block

        # OpenCV requires 3D array with channel dimension
        reshaped = as_strided(
            arr,
            shape=(height, width, bytes_per_pixel),
            strides=(stride, bytes_per_pixel, 1),
            writeable=False,
        )
    else:
        # Multi-plane formats (NV12, NV21)
        # OpenCV expects concatenated layout: (h * 3/2, w)
        reshaped = arr.reshape(height * 3 // 2, width)

    return cast(npt.NDArray[np.uint8], cv2.cvtColor(reshaped, cv_code))


def _convert_rgb(
    fmt: PixelFormat, width: int, height: int, stride: int, arr: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    cv_code = RGB_FORMAT_MAP[fmt.name]

    # Generic bytes_per_pixel from plane info
    plane = fmt.planes[0]
    bytes_per_pixel = plane.bytes_per_block // plane.pixels_per_block

    # OpenCV requires 3D array with channel dimension
    reshaped = as_strided(
        arr,
        shape=(height, width, bytes_per_pixel),
        strides=(stride, bytes_per_pixel, 1),
        writeable=False,
    )

    if cv_code is None:
        return reshaped.copy()

    return cast(npt.NDArray[np.uint8], cv2.cvtColor(reshaped, cv_code))


def _convert_raw(
    fmt: PixelFormat, width: int, height: int, stride: int, arr: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8] | None:
    pattern = fmt.bayer_pattern
    assert pattern is not None
    cv_code = BAYER_PATTERN_MAP[pattern]

    name = fmt.name
    plane = fmt.planes[0]

    # Determine element size from plane info
    bytes_per_pixel = plane.bytes_per_block // plane.pixels_per_block

    if bytes_per_pixel == 1:
        # 8-bit formats
        bayer = as_strided(arr, shape=(height, width), strides=(stride, 1), writeable=False)
        return cast(npt.NDArray[np.uint8], cv2.cvtColor(bayer, cv_code))
    elif bytes_per_pixel == 2:
        # 16-bit formats (could be 10, 12, or 16 bit stored in 16)
        arr16 = arr.view(np.uint16)
        bayer = as_strided(arr16, shape=(height, width), strides=(stride, 2), writeable=False)

        # Detect actual bit depth from format name for scaling
        if '16' in name:
            bgr16 = cast(npt.NDArray[np.uint16], cv2.cvtColor(bayer, cv_code))
            return (bgr16 >> 8).astype(np.uint8)
        elif '10' in name or '12' in name:
            bits = 10 if '10' in name else 12
            bayer = bayer << (16 - bits)
            bgr16 = cast(npt.NDArray[np.uint16], cv2.cvtColor(bayer, cv_code))
            return (bgr16 >> 8).astype(np.uint8)

    return None


def opencv_convert(
    fmt: PixelFormat, width: int, height: int, strides: tuple[int, ...], arr: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8] | None:
    bytesperline = strides[0]
    stride = bytesperline if bytesperline > 0 else fmt.stride(width, 0)

    if fmt.color == PixelColorEncoding.YUV:
        return _convert_yuv(fmt, width, height, stride, arr)

    if fmt.color == PixelColorEncoding.RAW:
        return _convert_raw(fmt, width, height, stride, arr)

    if fmt.color == PixelColorEncoding.RGB:
        return _convert_rgb(fmt, width, height, stride, arr)

    return None
