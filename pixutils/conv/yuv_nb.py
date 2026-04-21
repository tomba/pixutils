# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2025, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

"""Numba-optimized implementations for YUV pixel format conversions"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore[import-not-found]

from pixutils.conv.yuv import _get_conversion_matrix
from pixutils.formats import PixelFormat, PixelFormats

__all__ = ['yuv_to_bgr888_nb']


@njit(cache=True)
def _yuyv_to_bgr888_nb(
    data: npt.NDArray[np.uint8],
    width: int,
    height: int,
    stride: int,
    offset_y: float,
    offset_u: float,
    offset_v: float,
    m00: float,
    m01: float,
    m02: float,
    m10: float,
    m11: float,
    m12: float,
    m20: float,
    m21: float,
    m22: float,
) -> npt.NDArray[np.uint8]:
    """JIT-compiled YUYV to BGR conversion with direct pixel processing"""
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(0, width, 2):  # Process 2 pixels at a time
            # YUYV layout: Y0 U0 Y1 V0 (4 bytes for 2 pixels)
            base_idx = y * stride + x * 2

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
def _uyvy_to_bgr888_nb(
    data: npt.NDArray[np.uint8],
    width: int,
    height: int,
    stride: int,
    offset_y: float,
    offset_u: float,
    offset_v: float,
    m00: float,
    m01: float,
    m02: float,
    m10: float,
    m11: float,
    m12: float,
    m20: float,
    m21: float,
    m22: float,
) -> npt.NDArray[np.uint8]:
    """JIT-compiled UYVY to BGR conversion with direct pixel processing"""
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(0, width, 2):  # Process 2 pixels at a time
            # UYVY layout: U0 Y0 V0 Y1 (4 bytes for 2 pixels)
            base_idx = y * stride + x * 2

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
def _nv12_to_bgr888_nb(
    data: npt.NDArray[np.uint8],
    width: int,
    height: int,
    y_stride: int,
    uv_stride: int,
    offset_y: float,
    offset_u: float,
    offset_v: float,
    m00: float,
    m01: float,
    m02: float,
    m10: float,
    m11: float,
    m12: float,
    m20: float,
    m21: float,
    m22: float,
) -> npt.NDArray[np.uint8]:
    """JIT-compiled NV12 to BGR conversion with custom chroma upsampling"""
    rgb = np.empty((height, width, 3), dtype=np.uint8)

    # NV12 layout: Y plane followed by interleaved UV plane
    y_plane_offset = y_stride * height

    for y in range(height):
        for x in range(width):
            y_val = data[y * y_stride + x]

            # Get UV values from chroma plane (subsampled by 2x2)
            uv_y = y // 2
            uv_x = x // 2
            uv_idx = y_plane_offset + uv_y * uv_stride + uv_x * 2

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


def yuv_to_bgr888_nb(
    arr: npt.NDArray[np.uint8],
    w: int,
    h: int,
    strides: tuple[int, ...],
    fmt: PixelFormat,
    options: dict | None,
) -> npt.NDArray[np.uint8]:
    """Entry point for numba YUV conversions."""
    offset, matrix = _get_conversion_matrix(options)

    if fmt == PixelFormats.YUYV:
        return _yuyv_to_bgr888_nb(
            arr,
            w,
            h,
            strides[0],
            offset[0],
            offset[1],
            offset[2],
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        )

    if fmt == PixelFormats.UYVY:
        return _uyvy_to_bgr888_nb(
            arr,
            w,
            h,
            strides[0],
            offset[0],
            offset[1],
            offset[2],
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        )

    if fmt == PixelFormats.NV12:
        return _nv12_to_bgr888_nb(
            arr,
            w,
            h,
            strides[0],
            strides[1],
            offset[0],
            offset[1],
            offset[2],
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        )

    raise RuntimeError(f'Unsupported YUV format {fmt}')
