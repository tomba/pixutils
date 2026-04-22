# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat, PixelColorEncoding

from .yuv import yuv_to_bgr888
from .rgb import rgb_to_bgr888
from .raw import raw_to_bgr888
from .backends import get_backends


def to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int | Sequence[int],
    arr: npt.NDArray[np.uint8],
    options: None | dict = None,
) -> npt.NDArray[np.uint8]:
    """
    Convert a numpy array containing pixel data to BGR888 format.

    Parameters:
        fmt: The pixel format of the input data
        width: Width of the image in pixels
        height: Height of the image in pixels
        bytesperline: Bytes per line. Either:
            - 0: no padding, natural strides are used for every plane
            - a single non-zero int: stride of plane 0. For multiplane formats the
              strides of the other planes are extrapolated, preserving the padding
              ratio (matches libcamera convention)
            - a sequence of non-zero ints: one stride per plane
        arr: Numpy array containing the pixel data
        options: Optional dictionary with conversion options:
            - backends: List of backends in priority order, e.g. ['opencv', 'numba']
            - range: 'limited' or 'full' (for YUV formats)
            - encoding: 'bt601' (for YUV formats)
            - demosaic_method: '3x3', 'bilinear', 'mosaic', or 'opencv' (for RAW formats)

    Returns:
        Numpy array containing the image in BGR888 format
    """

    # Normalize bytesperline to a per-plane tuple of concrete (non-zero) strides
    if isinstance(bytesperline, int):
        if bytesperline == 0:
            strides = tuple(fmt.stride(width, i) for i in range(len(fmt.planes)))
        else:
            strides = tuple(fmt.extrapolate_stride(bytesperline, i) for i in range(len(fmt.planes)))
    else:
        if len(bytesperline) != len(fmt.planes):
            raise ValueError(
                f'Strides sequence length {len(bytesperline)} does not match number of planes {len(fmt.planes)}'
            )
        if any(s == 0 for s in bytesperline):
            raise ValueError('Strides sequence must contain non-zero stride for each plane')
        strides = tuple(bytesperline)

    # Get list of backends to try
    backends = get_backends(options.get('backends') if options else None)
    if not backends:
        raise ValueError('No backends available')

    size = 0

    for i in range(len(fmt.planes)):
        if strides[i] < fmt.stride(width, i):
            raise ValueError('bytesperline is too small')

        if arr.size < fmt.planesize(strides[i], height, i):
            raise ValueError(
                f'Input array is too small: {arr.size} < {fmt.planesize(strides[i], height, i)}, {bytesperline}, {strides}'
            )

        size += fmt.planesize(strides[i], height, i)

    # Get a view for the actual data
    arr = arr[:size]

    # Try backends in priority order
    result = None
    for backend in backends:
        if backend == 'opencv':
            from .opencv import opencv_to_bgr888

            result = opencv_to_bgr888(fmt, width, height, strides, arr, options)
            if result is not None:
                break
            # opencv couldn't handle this format/options, try next backend
            continue
        elif backend == 'numba':
            from .numba import numba_to_bgr888

            result = numba_to_bgr888(fmt, width, height, strides, arr, options)
            if result is not None:
                break
        elif backend == 'numpy':
            if fmt.color == PixelColorEncoding.YUV:
                result = yuv_to_bgr888(arr, width, height, strides, fmt, options)
            elif fmt.color == PixelColorEncoding.RAW:
                result = raw_to_bgr888(arr, width, height, strides, fmt, options)
            elif fmt.color == PixelColorEncoding.RGB:
                result = rgb_to_bgr888(fmt, width, height, strides, arr)
            else:
                raise ValueError(f'Unsupported format {fmt}')
            break

    if result is None:
        raise ValueError(f'No backend could handle {fmt.name} with given options')

    # Backends may return a view; guarantee only that it doesn't alias the
    # input buffer. Callers that need a specific layout can contiguify
    # themselves.
    if np.shares_memory(result, arr):
        result = result.copy()
    return result


def buffer_to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int | Sequence[int],
    buffer,
    options: None | dict = None,
) -> npt.NDArray[np.uint8]:
    """
    Convert a buffer-like object containing pixel data to BGR888 format.

    This function accepts any Buffer-like object, converts it to a numpy array,
    and then uses to_bgr888() to perform the conversion.

    Parameters:
        fmt: The pixel format of the input data
        width: Width of the image in pixels
        height: Height of the image in pixels
        bytesperline: Bytes per line. Either 0 (natural strides), a single
            non-zero int (stride of plane 0; other planes extrapolated), or a
            sequence of non-zero ints with one value per plane
        buffer: Buffer-like object containing the pixel data
        options: Optional dictionary with conversion options

    Returns:
        Numpy array containing the image in BGR888 format

    TODO:
        3.12+ supports collections.abc.Buffer which could be used for the input
        buffer
    """

    arr = np.frombuffer(buffer, dtype=np.uint8)
    rgb = to_bgr888(fmt, width, height, bytesperline, arr, options)
    return rgb
