# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2025, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

"""Numba-optimized implementations for raw pixel format conversions"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit # type: ignore[import-not-found]

from pixutils.formats import PixelFormat

# Import shared code from raw.py
from .raw import BayerPattern, RawFormat, prepare_unpacked_raw, mosaic

__all__ = ['raw_to_bgr888_nb']


@njit(cache=True)
def _unpack_10bit_nb(arr16: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """JIT-compiled 10-bit unpacking"""
    height, width = arr16.shape
    result = np.empty((height, width * 4 // 5), dtype=np.uint16)

    for row in range(height):
        for col_group in range(width // 5):
            base_col = col_group * 5
            out_col = col_group * 4

            # Extract the 4 pixels + 1 byte of packed data
            p0 = arr16[row, base_col + 0] << 2
            p1 = arr16[row, base_col + 1] << 2
            p2 = arr16[row, base_col + 2] << 2
            p3 = arr16[row, base_col + 3] << 2
            packed = arr16[row, base_col + 4]

            # Distribute the 2 LSBs from packed byte
            result[row, out_col + 0] = p0 | ((packed >> 6) & 0b11)
            result[row, out_col + 1] = p1 | ((packed >> 4) & 0b11)
            result[row, out_col + 2] = p2 | ((packed >> 2) & 0b11)
            result[row, out_col + 3] = p3 | ((packed >> 0) & 0b11)

    return result


@njit(cache=True)
def _unpack_12bit_nb(arr16: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """JIT-compiled 12-bit unpacking"""
    height, width = arr16.shape
    result = np.empty((height, width * 2 // 3), dtype=np.uint16)

    for row in range(height):
        for col_group in range(width // 3):
            base_col = col_group * 3
            out_col = col_group * 2

            # Extract the 2 pixels + 1 byte of packed data
            p0 = arr16[row, base_col + 0] << 4
            p1 = arr16[row, base_col + 1] << 4
            packed = arr16[row, base_col + 2]

            # Distribute the 4 LSBs from packed byte
            result[row, out_col + 0] = p0 | ((packed >> 4) & 0b1111)
            result[row, out_col + 1] = p1 | ((packed >> 0) & 0b1111)

    return result


@njit(parallel=True, cache=True)
def _demosaic_bilinear_nb(data: npt.NDArray[np.uint16], r0, g0, g1, b0,
                          h: int, w: int) -> npt.NDArray[np.uint16]:
    """JIT-compiled bilinear interpolation demosaic - structured processing"""
    output = np.zeros((h, w, 3), dtype=data.dtype)

    # Extract Bayer pattern positions
    r_y, r_x = r0
    g0_y, g0_x = g0
    g1_y, g1_x = g1
    b_y, b_x = b0

    # Copy known pixel values to their respective channels
    for y in range(r_y, h, 2):
        for x in range(r_x, w, 2):
            output[y, x, 0] = data[y, x]  # Red

    for y in range(g0_y, h, 2):
        for x in range(g0_x, w, 2):
            output[y, x, 1] = data[y, x]  # Green

    for y in range(g1_y, h, 2):
        for x in range(g1_x, w, 2):
            output[y, x, 1] = data[y, x]  # Green

    for y in range(b_y, h, 2):
        for x in range(b_x, w, 2):
            output[y, x, 2] = data[y, x]  # Blue

    # Process Red positions - interpolate Green and Blue
    for y in range(r_y, h, 2):
        for x in range(r_x, w, 2):
            # Interpolate Green from cross neighbors
            g_sum = 0
            g_count = 0
            if y > 0:
                g_sum += output[y-1, x, 1]
                g_count += 1
            if y < h-1:
                g_sum += output[y+1, x, 1]
                g_count += 1
            if x > 0:
                g_sum += output[y, x-1, 1]
                g_count += 1
            if x < w-1:
                g_sum += output[y, x+1, 1]
                g_count += 1
            if g_count > 0:
                output[y, x, 1] = g_sum // g_count

            # Interpolate Blue from diagonal neighbors
            b_sum = 0
            b_count = 0
            if y > 0 and x > 0:
                b_sum += output[y-1, x-1, 2]
                b_count += 1
            if y > 0 and x < w-1:
                b_sum += output[y-1, x+1, 2]
                b_count += 1
            if y < h-1 and x > 0:
                b_sum += output[y+1, x-1, 2]
                b_count += 1
            if y < h-1 and x < w-1:
                b_sum += output[y+1, x+1, 2]
                b_count += 1
            if b_count > 0:
                output[y, x, 2] = b_sum // b_count

    # Process Blue positions - interpolate Red and Green
    for y in range(b_y, h, 2):
        for x in range(b_x, w, 2):
            # Interpolate Red from diagonal neighbors
            r_sum = 0
            r_count = 0
            if y > 0 and x > 0:
                r_sum += output[y-1, x-1, 0]
                r_count += 1
            if y > 0 and x < w-1:
                r_sum += output[y-1, x+1, 0]
                r_count += 1
            if y < h-1 and x > 0:
                r_sum += output[y+1, x-1, 0]
                r_count += 1
            if y < h-1 and x < w-1:
                r_sum += output[y+1, x+1, 0]
                r_count += 1
            if r_count > 0:
                output[y, x, 0] = r_sum // r_count

            # Interpolate Green from cross neighbors
            g_sum = 0
            g_count = 0
            if y > 0:
                g_sum += output[y-1, x, 1]
                g_count += 1
            if y < h-1:
                g_sum += output[y+1, x, 1]
                g_count += 1
            if x > 0:
                g_sum += output[y, x-1, 1]
                g_count += 1
            if x < w-1:
                g_sum += output[y, x+1, 1]
                g_count += 1
            if g_count > 0:
                output[y, x, 1] = g_sum // g_count

    # Process Green positions (both G0 and G1) - interpolate Red and Blue
    for y in range(g0_y, h, 2):
        for x in range(g0_x, w, 2):
            # Interpolate Red from horizontal neighbors
            r_sum = 0
            r_count = 0
            if x > 0:
                r_sum += output[y, x-1, 0]
                r_count += 1
            if x < w-1:
                r_sum += output[y, x+1, 0]
                r_count += 1
            if r_count > 0:
                output[y, x, 0] = r_sum // r_count

            # Interpolate Blue from vertical neighbors
            b_sum = 0
            b_count = 0
            if y > 0:
                b_sum += output[y-1, x, 2]
                b_count += 1
            if y < h-1:
                b_sum += output[y+1, x, 2]
                b_count += 1
            if b_count > 0:
                output[y, x, 2] = b_sum // b_count

    for y in range(g1_y, h, 2):
        for x in range(g1_x, w, 2):
            # Interpolate Blue from horizontal neighbors
            b_sum = 0
            b_count = 0
            if x > 0:
                b_sum += output[y, x-1, 2]
                b_count += 1
            if x < w-1:
                b_sum += output[y, x+1, 2]
                b_count += 1
            if b_count > 0:
                output[y, x, 2] = b_sum // b_count

            # Interpolate Red from vertical neighbors
            r_sum = 0
            r_count = 0
            if y > 0:
                r_sum += output[y-1, x, 0]
                r_count += 1
            if y < h-1:
                r_sum += output[y+1, x, 0]
                r_count += 1
            if r_count > 0:
                output[y, x, 0] = r_sum // r_count

    return output


@njit(parallel=True, cache=True)
def _compute_demosaic_planes_nb(rgb: npt.NDArray[np.uint16], bayer: npt.NDArray[np.uint8],
                                output_height: int, output_width: int) -> npt.NDArray[np.uint16]:
    """JIT-compiled function to compute the demosaic for all RGB planes"""
    output = np.empty((output_height, output_width, 3), dtype=rgb.dtype)

    for plane in range(3):
        p = rgb[..., plane].astype(np.uint32)
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


def _prepare_packed_raw_nb(data: npt.NDArray[np.uint8], width: int, height: int,
                           bits_per_pixel: int, bytesperline: int) -> npt.NDArray[np.uint16]:
    """Prepare packed raw data using numba unpacking."""
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

    # Unpack to 16-bit using numba functions
    arr16_input = data.astype(np.uint16)
    if bits_per_pixel == 10:
        return _unpack_10bit_nb(arr16_input)
    else:  # 12-bit
        return _unpack_12bit_nb(arr16_input)


def _demosaic_3x3_window_nb(data: npt.NDArray[np.uint16], pattern: BayerPattern,
                            h: int, w: int) -> npt.NDArray[np.uint16]:
    """3x3 window demosaic using numba."""
    # Separate the components from the Bayer data to RGB planes
    rgb = np.zeros((h, w, 3), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[pattern.r0[1] :: 2, pattern.r0[0] :: 2]  # Red
    rgb[0::2, 0::2, 1] = data[pattern.g0[1] :: 2, pattern.g0[0] :: 2]  # Green
    rgb[1::2, 1::2, 1] = data[pattern.g1[1] :: 2, pattern.g1[0] :: 2]  # Green
    rgb[0::2, 1::2, 2] = data[pattern.b0[1] :: 2, pattern.b0[0] :: 2]  # Blue

    bayer = np.zeros(rgb.shape, dtype=np.uint8)
    bayer[1::2, 0::2, 0] = 1  # Red
    bayer[0::2, 0::2, 1] = 1  # Green
    bayer[1::2, 1::2, 1] = 1  # Green
    bayer[0::2, 1::2, 2] = 1  # Blue

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

    return _compute_demosaic_planes_nb(rgb, bayer, h, w)


def _demosaic_nb(data: npt.NDArray[np.uint16], pattern: BayerPattern,
                 options: None | dict = None) -> npt.NDArray[np.uint16]:
    """Demosaic using numba implementations."""
    method = options.get('demosaic_method', '3x3') if options else '3x3'
    h, w = data.shape

    if method == 'mosaic':
        return mosaic(data, pattern)
    elif method == 'bilinear':
        return _demosaic_bilinear_nb(data, pattern.r0, pattern.g0, pattern.g1, pattern.b0, h, w)
    elif method == '3x3':
        return _demosaic_3x3_window_nb(data, pattern, h, w)
    else:
        raise ValueError(f'Unknown demosaic method: {method}')


def raw_to_bgr888_nb(data: npt.NDArray[np.uint8], width: int, height: int,
                     bytesperline: int, fmt: PixelFormat,
                     options: None | dict = None) -> npt.NDArray[np.uint8]:
    """Entry point for numba RAW conversions."""
    # Parse the format
    raw_fmt = RawFormat.from_pixelformat(fmt)

    # Prepare the raw data into a common 16-bit format
    if raw_fmt.is_packed:
        arr16 = _prepare_packed_raw_nb(data, width, height, raw_fmt.bits_per_pixel,
                                       bytesperline)
    else:
        arr16 = prepare_unpacked_raw(data, width, height, raw_fmt.bits_per_pixel,
                                     bytesperline)

    # Perform demosaic
    rgb = _demosaic_nb(arr16, raw_fmt.bayer_pattern, options)

    # Convert to 8-bit BGR
    return (rgb >> (raw_fmt.bits_per_pixel - 8)).astype(np.uint8)
