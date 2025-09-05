# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2025, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

"""Numba-optimized implementations for raw pixel format conversions"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def unpack_10bit_nb(arr16: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
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
def unpack_12bit_nb(arr16: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
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
def _demosaic_bilinear_nb(data: npt.NDArray[np.uint16], r0, g0, g1, b0, h: int, w: int) -> npt.NDArray[np.uint16]:
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
def compute_demosaic_planes_nb(rgb: npt.NDArray[np.uint16], bayer: npt.NDArray[np.uint8],
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
