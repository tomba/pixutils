# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2025, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

"""Numba-optimized implementations for YUV pixel format conversions"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit



@njit(cache=True)
def yuyv_to_bgr888_nb(data: npt.NDArray[np.uint8], width: int, height: int,
                      offset_y: float, offset_u: float, offset_v: float,
                      m00: float, m01: float, m02: float,
                      m10: float, m11: float, m12: float,
                      m20: float, m21: float, m22: float) -> npt.NDArray[np.uint8]:
    """JIT-compiled YUYV to BGR conversion with direct pixel processing"""
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(0, width, 2):  # Process 2 pixels at a time
            # YUYV layout: Y0 U0 Y1 V0 (4 bytes for 2 pixels)
            # Each row has width*2 bytes, each pair of pixels needs 4 bytes
            base_idx = y * width * 2 + x * 2

            y0 = data[base_idx + 0]
            u = data[base_idx + 1]
            y1 = data[base_idx + 2]
            v = data[base_idx + 3]

            # Process both pixels with shared chroma
            for px in range(2):
                if x + px >= width:
                    break

                y_val = y0 if px == 0 else y1

                # Apply offsets
                y_adj = y_val + offset_y
                u_adj = u + offset_u
                v_adj = v + offset_v

                # Matrix multiplication: [Y U V] × Matrix (column-wise produces BGR)
                b = m00 * y_adj + m10 * u_adj + m20 * v_adj
                g = m01 * y_adj + m11 * u_adj + m21 * v_adj
                r = m02 * y_adj + m12 * u_adj + m22 * v_adj

                # Clip and store as BGR
                rgb[y, x + px, 0] = max(0, min(255, int(b)))  # B
                rgb[y, x + px, 1] = max(0, min(255, int(g)))  # G
                rgb[y, x + px, 2] = max(0, min(255, int(r)))  # R

    return rgb


@njit(cache=True)
def uyvy_to_bgr888_nb(data: npt.NDArray[np.uint8], width: int, height: int,
                      offset_y: float, offset_u: float, offset_v: float,
                      m00: float, m01: float, m02: float,
                      m10: float, m11: float, m12: float,
                      m20: float, m21: float, m22: float) -> npt.NDArray[np.uint8]:
    """JIT-compiled UYVY to BGR conversion with direct pixel processing"""
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(0, width, 2):  # Process 2 pixels at a time
            # UYVY layout: U0 Y0 V0 Y1 (4 bytes for 2 pixels)
            # Each row has width*2 bytes, each pair of pixels needs 4 bytes
            base_idx = y * width * 2 + x * 2

            u = data[base_idx + 0]
            y0 = data[base_idx + 1]
            v = data[base_idx + 2]
            y1 = data[base_idx + 3]

            # Process both pixels with shared chroma
            for px in range(2):
                if x + px >= width:
                    break

                y_val = y0 if px == 0 else y1

                # Apply offsets
                y_adj = y_val + offset_y
                u_adj = u + offset_u
                v_adj = v + offset_v

                # Matrix multiplication: [Y U V] × Matrix (column-wise produces BGR)
                b = m00 * y_adj + m10 * u_adj + m20 * v_adj
                g = m01 * y_adj + m11 * u_adj + m21 * v_adj
                r = m02 * y_adj + m12 * u_adj + m22 * v_adj

                # Clip and store as BGR
                rgb[y, x + px, 0] = max(0, min(255, int(b)))  # B
                rgb[y, x + px, 1] = max(0, min(255, int(g)))  # G
                rgb[y, x + px, 2] = max(0, min(255, int(r)))  # R

    return rgb


@njit(cache=True)
def nv12_to_bgr888_nb(data: npt.NDArray[np.uint8], width: int, height: int,
                      offset_y: float, offset_u: float, offset_v: float,
                      m00: float, m01: float, m02: float,
                      m10: float, m11: float, m12: float,
                      m20: float, m21: float, m22: float) -> npt.NDArray[np.uint8]:
    """JIT-compiled NV12 to BGR conversion with custom chroma upsampling"""
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    # NV12 layout: Y plane followed by interleaved UV plane
    y_plane_size = width * height

    for y in range(height):
        for x in range(width):
            # Get Y value directly
            y_val = data[y * width + x]

            # Get UV values from chroma plane (subsampled by 2x2)
            uv_y = y // 2
            uv_x = x // 2
            uv_idx = y_plane_size + uv_y * width + uv_x * 2

            u = data[uv_idx + 0]
            v = data[uv_idx + 1]

            # Apply offsets
            y_adj = y_val + offset_y
            u_adj = u + offset_u
            v_adj = v + offset_v

            # Matrix multiplication: [Y U V] × Matrix (column-wise produces BGR)
            b = m00 * y_adj + m10 * u_adj + m20 * v_adj
            g = m01 * y_adj + m11 * u_adj + m21 * v_adj
            r = m02 * y_adj + m12 * u_adj + m22 * v_adj

            # Clip and store as BGR
            rgb[y, x, 0] = max(0, min(255, int(b)))  # B
            rgb[y, x, 1] = max(0, min(255, int(g)))  # G
            rgb[y, x, 2] = max(0, min(255, int(r)))  # R

    return rgb
