# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat

# Try to import numba-optimized functions
if os.environ.get('PIXUTILS_DISABLE_NUMBA'):
    USE_NUMBA = False
else:
    try:
        from .raw_nb import unpack_10bit_nb, unpack_12bit_nb, _demosaic_bilinear_nb, compute_demosaic_planes_nb
        USE_NUMBA = True
    except ImportError:
        USE_NUMBA = False

__all__ = ['raw_to_bgr888']


@dataclass
class BayerPattern:
    """Represents a Bayer pattern configuration"""

    r0: tuple[int, int]
    g0: tuple[int, int]
    g1: tuple[int, int]
    b0: tuple[int, int]

    @classmethod
    def from_pattern(cls, pattern: str):
        """Parse a Bayer pattern string (e.g., 'SRGGB') into coordinates"""
        idx = pattern.find('R')
        r0 = (idx % 2, idx // 2)

        idx = pattern.find('G')
        g0 = (idx % 2, idx // 2)

        idx = pattern.find('G', idx + 1)
        g1 = (idx % 2, idx // 2)

        idx = pattern.find('B')
        b0 = (idx % 2, idx // 2)

        return cls(r0, g0, g1, b0)


@dataclass
class RawFormat:
    """Represents a raw image format configuration"""

    bayer_pattern: BayerPattern
    bits_per_pixel: int
    is_packed: bool

    @classmethod
    def from_pixelformat(cls, fmt: PixelFormat):
        """Parse a PixelFormat into raw format configuration"""
        name = fmt.name
        pattern = name[1:5]  # e.g., 'RGGB' from 'SRGGB8'
        is_packed = name.endswith('P')

        if is_packed:
            bpp = int(name[5:-1])  # Remove 'P' for packed formats
        else:
            bpp = int(name[5:])  # Direct BPP value

        return cls(bayer_pattern=BayerPattern.from_pattern(pattern),
                   bits_per_pixel=bpp,
                   is_packed=is_packed)


def prepare_packed_raw(data: npt.NDArray[np.uint8], width: int, height: int,
                      bits_per_pixel: int, bytesperline: int) -> npt.NDArray[np.uint16]:
    assert bits_per_pixel in [10, 12], 'Only 10 and 12 bpp are supported'

    # Reshape into rows if bytesperline is provided
    if bytesperline:
        data = data.reshape((len(data) // bytesperline, bytesperline))
    else:
        data = data.reshape((height, len(data) // height))

    # Remove padding if present
    padded_width = width * bits_per_pixel // 8
    if bytesperline > padded_width:
        data = np.delete(data, np.s_[padded_width:], 1)

    # Unpack to 16-bit
    arr16_input = data.astype(np.uint16)
    if bits_per_pixel == 10:
        if USE_NUMBA:
            arr16 = unpack_10bit_nb(arr16_input)  # type: ignore[possibly-undefined]
        else:
            arr16 = _unpack_10bit(arr16_input)
    else:  # 12-bit
        if USE_NUMBA:
            arr16 = unpack_12bit_nb(arr16_input)  # type: ignore[possibly-undefined]
        else:
            arr16 = _unpack_12bit(arr16_input)

    return arr16


def _unpack_10bit(arr16: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Pure numpy implementation for 10-bit unpacking"""
    arr16_shifted = arr16 << np.uint16(2)
    for byte in range(4):
        arr16_shifted[:, byte::5] |= (arr16_shifted[:, 4::5] >> ((4 - byte) * 2)) & 0b11
    return np.delete(arr16_shifted, np.s_[4::5], 1)


def _unpack_12bit(arr16: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Pure numpy implementation for 12-bit unpacking"""
    arr16_shifted = arr16 << np.uint16(4)
    for byte in range(2):
        arr16_shifted[:, byte::3] |= (arr16_shifted[:, 2::3] >> ((2 - byte) * 4)) & 0b1111
    return np.delete(arr16_shifted, np.s_[2::3], 1)


def prepare_unpacked_raw(data: npt.NDArray[np.uint8], width: int, height: int,
                        bits_per_pixel: int, bytesperline: int) -> npt.NDArray[np.uint16]:

    # Reshape into rows if bytesperline is provided
    if bytesperline:
        data = data.reshape((len(data) // bytesperline, bytesperline))
    else:
        data = data.reshape((height, len(data) // height))

    # Remove padding if present.
    # The unpacked data is stored in 8 bits for 8bpp, and 16 bits for 10/12/16bpp.
    bytes_per_pixel = (bits_per_pixel + 7) // 8
    padded_width = width * bytes_per_pixel
    if bytesperline > padded_width:
        data = np.delete(data, np.s_[padded_width:], 1)

    # Expand 8 bit data into 16 bit containers
    if bits_per_pixel == 8:
        return data.reshape((height, width)).astype(np.uint16)

    if bits_per_pixel in [10, 12, 16]:
        return data.view(np.uint16).reshape((height, width))

    raise RuntimeError(f'Unsupported bits per pixel: {bits_per_pixel}')


def demosaic(data: npt.NDArray[np.uint16], pattern: BayerPattern, options: None | dict = None) -> npt.NDArray[np.uint16]:
    # Select demosaic algorithm based on options
    method = options.get('demosaic_method', '3x3') if options else '3x3'
    h, w = data.shape

    if method == 'bilinear':
        if USE_NUMBA:
            return _demosaic_bilinear_nb(data, pattern.r0, pattern.g0, pattern.g1, pattern.b0, h, w)  # type: ignore[possibly-undefined]
        else:
            raise NotImplementedError('Bilinear demosaic not available without Numba')
    elif method == '3x3':
        return _demosaic_3x3_window(data, pattern, h, w)
    else:
        raise ValueError(f'Unknown demosaic method: {method}')


def _demosaic_3x3_window(data: npt.NDArray[np.uint16], pattern: BayerPattern, h: int, w: int) -> npt.NDArray[np.uint16]:
    """3x3 window demosaic algorithm with automatic Numba/Python backend selection"""
    # Separate the components from the Bayer data to RGB planes
    rgb = np.zeros((h, w, 3), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[pattern.r0[1] :: 2, pattern.r0[0] :: 2]  # Red
    rgb[0::2, 0::2, 1] = data[pattern.g0[1] :: 2, pattern.g0[0] :: 2]  # Green
    rgb[1::2, 1::2, 1] = data[pattern.g1[1] :: 2, pattern.g1[0] :: 2]  # Green
    rgb[0::2, 1::2, 2] = data[pattern.b0[1] :: 2, pattern.b0[0] :: 2]  # Blue

    # Below we present a fairly naive de-mosaic method that simply
    # calculates the weighted average of a pixel based on the pixels
    # surrounding it. The weighting is provided by a byte representation of
    # the Bayer filter which we construct first:

    bayer = np.zeros(rgb.shape, dtype=np.uint8)
    bayer[1::2, 0::2, 0] = 1  # Red
    bayer[0::2, 0::2, 1] = 1  # Green
    bayer[1::2, 1::2, 1] = 1  # Green
    bayer[0::2, 1::2, 2] = 1  # Blue

    # Allocate an array to hold our output with the same shape as the input
    # data. After this we define the size of window that will be used to
    # calculate each weighted average (3x3). Then we pad out the rgb and
    # bayer arrays, adding blank pixels at their edges to compensate for the
    # size of the window when calculating averages for edge pixels.

    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)

    rgb = np.pad(rgb, [
        (border[0], border[0]),
        (border[1], border[1]),
        (0, 0),
    ], 'constant')
    bayer = np.pad(bayer, [
        (border[0], border[0]),
        (border[1], border[1]),
        (0, 0),
    ], 'constant')

    # Choose backend based on Numba availability
    if USE_NUMBA:
        return compute_demosaic_planes_nb(rgb, bayer, h, w)  # type: ignore[possibly-undefined]
    else:
        return _compute_demosaic_planes(rgb, bayer, h, w)


def _compute_demosaic_planes(rgb: npt.NDArray[np.uint16], bayer: npt.NDArray[np.uint8],
                            output_height: int, output_width: int) -> npt.NDArray[np.uint16]:
    # For each plane in the RGB data, we calculate the 3x3 window sum
    # and divide it with the weighted average. This version uses direct
    # computation of the sum, instead of using numpy's as_strided()
    # and einsum(), as the direct version is 2x as fast.

    output = np.empty((output_height, output_width, 3), dtype=rgb.dtype)

    for plane in range(3):
        p = rgb[..., plane].astype(np.uint32, copy=False)
        b = bayer[..., plane]

        # Direct computation of 3x3 window sum
        psum = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])

        bsum = (b[:-2, :-2] + b[:-2, 1:-1] + b[:-2, 2:] +
                b[1:-1, :-2] + b[1:-1, 1:-1] + b[1:-1, 2:] +
                b[2:, :-2] + b[2:, 1:-1] + b[2:, 2:])

        output[..., plane] = psum // bsum

    return output


def raw_to_bgr888(data: npt.NDArray[np.uint8], width: int, height: int,
                  bytesperline: int, fmt: PixelFormat,
                  options: None | dict = None) -> npt.NDArray[np.uint8]:
    # Parse the format
    raw_fmt = RawFormat.from_pixelformat(fmt)

    # Prepare the raw data into a common 16-bit format
    if raw_fmt.is_packed:
        arr16 = prepare_packed_raw(data, width, height, raw_fmt.bits_per_pixel,
                                   bytesperline)
    else:
        arr16 = prepare_unpacked_raw(data, width, height, raw_fmt.bits_per_pixel,
                                     bytesperline)

    # Perform demosaic
    rgb = demosaic(arr16, raw_fmt.bayer_pattern, options)

    # Convert to 8-bit BGR
    return (rgb >> (raw_fmt.bits_per_pixel - 8)).astype(np.uint8)
