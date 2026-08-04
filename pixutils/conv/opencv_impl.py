# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from typing import cast

import cv2  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided

from pixutils.formats import PixelColorEncoding, PixelFormat, PixelFormats

from .utils import strip_padding

__all__ = ['opencv_convert']

BAYER_PATTERN_MAP: dict[str, int] = {
    'RGGB': cv2.COLOR_BAYER_RG2BGR,
    'BGGR': cv2.COLOR_BAYER_BG2BGR,
    'GRBG': cv2.COLOR_BAYER_GR2BGR,
    'GBRG': cv2.COLOR_BAYER_GB2BGR,
}

YUV_FORMAT_MAP: dict[str, int] = {
    'YUYV': cv2.COLOR_YUV2RGB_YUY2,
    'UYVY': cv2.COLOR_YUV2RGB_UYVY,
    'YVYU': cv2.COLOR_YUV2RGB_YVYU,
    'NV12': cv2.COLOR_YUV2RGB_NV12,
    'NV21': cv2.COLOR_YUV2RGB_NV21,
}


def _convert_yuv(
    fmt: PixelFormat, width: int, height: int, strides: tuple[int, ...], arr: npt.NDArray[np.uint8]
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
            strides=(strides[0], bytes_per_pixel, 1),
            writeable=False,
        )
    else:
        # Multi-plane formats (NV12, NV21)
        # OpenCV expects concatenated layout: (h * 3/2, w)
        arr = strip_padding(arr, height, strides, fmt, width)
        reshaped = arr.reshape(height * 3 // 2, width)

    return cast(npt.NDArray[np.uint8], cv2.cvtColor(reshaped, cv_code))


def _convert_rgb(
    fmt: PixelFormat, width: int, height: int, stride: int, arr: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    plane = fmt.planes[0]
    bytes_per_pixel = plane.bytes_per_block // plane.pixels_per_block

    reshaped = as_strided(
        arr,
        shape=(height, width, bytes_per_pixel),
        strides=(stride, bytes_per_pixel, 1),
        writeable=False,
    )

    # Note: OpenCV uses reverse channel order naming than pixutils
    if fmt == PixelFormats.BGR888:
        result = reshaped
    elif fmt == PixelFormats.RGB888:
        result = cv2.cvtColor(reshaped, cv2.COLOR_BGR2RGB)
    elif fmt in (PixelFormats.XRGB8888, PixelFormats.ARGB8888):
        result = cv2.cvtColor(reshaped, cv2.COLOR_BGRA2RGB)
    elif fmt in (PixelFormats.XBGR8888, PixelFormats.ABGR8888):
        result = cv2.cvtColor(reshaped, cv2.COLOR_RGBA2RGB)
    elif fmt in (PixelFormats.RGBX8888, PixelFormats.RGBA8888):
        # Rotate X/A to the end
        rotated = reshaped[..., [1, 2, 3, 0]]
        result = cv2.cvtColor(rotated, cv2.COLOR_BGRA2RGB)
    elif fmt in (PixelFormats.BGRX8888, PixelFormats.BGRA8888):
        # Rotate X/A to the end
        rotated = reshaped[..., [1, 2, 3, 0]]
        result = cv2.cvtColor(rotated, cv2.COLOR_RGBA2RGB)
    else:
        raise NotImplementedError(f'Unsupported RGB format {fmt.name}')

    return cast(npt.NDArray[np.uint8], result)


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
    if fmt.color == PixelColorEncoding.YUV:
        return _convert_yuv(fmt, width, height, strides, arr)

    if fmt.color == PixelColorEncoding.RAW:
        return _convert_raw(fmt, width, height, strides[0], arr)

    if fmt.color == PixelColorEncoding.RGB:
        return _convert_rgb(fmt, width, height, strides[0], arr)

    return None
