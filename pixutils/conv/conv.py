# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from collections.abc import Buffer

from pixutils.formats import PixelFormat, PixelColorEncoding

from .yuv import yuv_to_bgr888
from .rgb import rgb_to_bgr888
from .raw import raw_to_bgr888


def to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int,
    arr: npt.NDArray[np.uint8],
    options: None | dict = None,
):
    """
    Convert a numpy array containing pixel data to BGR888 format.

    Parameters:
        fmt: The pixel format of the input data
        width: Width of the image in pixels
        height: Height of the image in pixels
        bytesperline: Number of bytes per line in the input data or 0 for no padding
        arr: Numpy array containing the pixel data
        options: Optional dictionary with conversion options

    Returns:
        Numpy array containing the image in BGR888 format
    """

    if fmt.color == PixelColorEncoding.YUV:
        return yuv_to_bgr888(arr, width, height, fmt, options)

    if fmt.color == PixelColorEncoding.RAW:
        return raw_to_bgr888(arr, width, height, bytesperline, fmt)

    if fmt.color == PixelColorEncoding.RGB:
        return rgb_to_bgr888(fmt, width, height, arr)

    raise RuntimeError(f'Unsupported format {fmt}')


def buffer_to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int,
    buffer: Buffer,
    options: None | dict = None,
):
    """
    Convert a buffer-like object containing pixel data to BGR888 format.

    This function accepts any Buffer-like object, converts it to a numpy array,
    and then uses to_bgr888() to perform the conversion.

    Parameters:
        fmt: The pixel format of the input data
        width: Width of the image in pixels
        height: Height of the image in pixels
        bytesperline: Number of bytes per line in the input data or 0 for no padding
        buffer: Buffer-like object containing the pixel data
        options: Optional dictionary with conversion options

    Returns:
        Numpy array containing the image in BGR888 format
    """

    arr = np.frombuffer(buffer, dtype=np.uint8)
    rgb = to_bgr888(fmt, width, height, bytesperline, arr, options)
    return rgb
