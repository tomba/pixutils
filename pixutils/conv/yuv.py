# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided

from pixutils.formats import PixelFormat, PixelFormats

# Generated with './utils/gen-csc.py --format python --transpose'

YCBCR_VALUES = {
    'bt601': {
        'full': {
            'offsets': (0, -128, -128),
            'matrix': [
                [1.00000000, 1.00000000, 1.00000000],
                [0.00000000, -0.34413629, 1.77200000],
                [1.40200000, -0.71413629, 0.00000000],
            ],
        },
        'limited': {
            'offsets': (-16, -128, -128),
            'matrix': [
                [1.16438356, 1.16438356, 1.16438356],
                [0.00000000, -0.39176229, 2.01723214],
                [1.59602679, -0.81296765, 0.00000000],
            ],
        },
    },
    'bt709': {
        'full': {
            'offsets': (0, -128, -128),
            'matrix': [
                [1.00000000, 1.00000000, 1.00000000],
                [0.00000000, -0.18732427, 1.85560000],
                [1.57480000, -0.46812427, 0.00000000],
            ],
        },
        'limited': {
            'offsets': (-16, -128, -128),
            'matrix': [
                [1.16438356, 1.16438356, 1.16438356],
                [0.00000000, -0.21324861, 2.11240179],
                [1.79274107, -0.53290933, 0.00000000],
            ],
        },
    },
    'bt2020': {
        'full': {
            'offsets': (0, -128, -128),
            'matrix': [
                [1.00000000, 1.00000000, 1.00000000],
                [0.00000000, -0.16455313, 1.88140000],
                [1.47460000, -0.57135313, 0.00000000],
            ],
        },
        'limited': {
            'offsets': (-16, -128, -128),
            'matrix': [
                [1.16438356, 1.16438356, 1.16438356],
                [0.00000000, -0.18732610, 2.14177232],
                [1.67867411, -0.65042432, 0.00000000],
            ],
        },
    },
}


def _get_conversion_matrix(options: dict | None) -> tuple[tuple[int, int, int], list[list[float]]]:
    """Get color conversion offset and matrix from options"""
    color_range = 'limited'
    color_encoding = 'bt601'

    if options:
        color_range = options.get('range', color_range)
        color_encoding = options.get('encoding', color_encoding)

    conv_data = YCBCR_VALUES[color_encoding][color_range]
    return conv_data['offsets'], conv_data['matrix']


def ycbcr_to_bgr888(yuv: npt.NDArray[np.uint8], options: dict | None) -> npt.NDArray[np.uint8]:
    offset, matrix = _get_conversion_matrix(options)

    offset = np.array(offset)
    m = np.array(matrix)

    rgb = np.dot(yuv + offset, m)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)


def yuyv_to_bgr888(
    data: npt.NDArray[np.uint8], w: int, h: int, stride: int, options: dict | None
) -> npt.NDArray[np.uint8]:
    yuyv = as_strided(data, shape=(h, w // 2 * 4), strides=(stride, 1), writeable=False)

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = yuyv[:, 0::2]  # Y
    yuv[:, :, 1] = yuyv[:, 1::4].repeat(2, axis=1)  # U
    yuv[:, :, 2] = yuyv[:, 3::4].repeat(2, axis=1)  # V

    return ycbcr_to_bgr888(yuv, options)


def uyvy_to_bgr888(
    data: npt.NDArray[np.uint8], w: int, h: int, stride: int, options: dict | None
) -> npt.NDArray[np.uint8]:
    yuyv = as_strided(data, shape=(h, w // 2 * 4), strides=(stride, 1), writeable=False)

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = yuyv[:, 1::2]  # Y
    yuv[:, :, 1] = yuyv[:, 0::4].repeat(2, axis=1)  # U
    yuv[:, :, 2] = yuyv[:, 2::4].repeat(2, axis=1)  # V

    return ycbcr_to_bgr888(yuv, options)


def nv_to_bgr888(
    data: npt.NDArray[np.uint8],
    w: int,
    h: int,
    y_stride: int,
    uv_stride: int,
    v_subsample: int,
    u_first: bool,
    options: dict | None,
) -> npt.NDArray[np.uint8]:
    y = as_strided(data, shape=(h, w), strides=(y_stride, 1), writeable=False)
    uv = as_strided(
        data[y_stride * h :],
        shape=(h // v_subsample, w // 2, 2),
        strides=(uv_stride, 2, 1),
        writeable=False,
    )

    u_idx = 0 if u_first else 1
    v_idx = 1 - u_idx

    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = y

    u = uv[:, :, u_idx]
    v = uv[:, :, v_idx]
    if v_subsample == 2:
        u = u.repeat(2, axis=0)
        v = v.repeat(2, axis=0)
    yuv[:, :, 1] = u.repeat(2, axis=1)
    yuv[:, :, 2] = v.repeat(2, axis=1)

    return ycbcr_to_bgr888(yuv, options)


def planar_yuv_to_bgr888(
    data: npt.NDArray[np.uint8],
    w: int,
    h: int,
    y_stride: int,
    c1_stride: int,
    c2_stride: int,
    v_subsample: int,
    u_first: bool,
    options: dict | None,
) -> npt.NDArray[np.uint8]:
    c_h = h // v_subsample
    c_w = w // 2

    y_size = y_stride * h
    c1_size = c1_stride * c_h

    y = as_strided(data, shape=(h, w), strides=(y_stride, 1), writeable=False)
    c1 = as_strided(data[y_size:], shape=(c_h, c_w), strides=(c1_stride, 1), writeable=False)
    c2 = as_strided(
        data[y_size + c1_size :], shape=(c_h, c_w), strides=(c2_stride, 1), writeable=False
    )

    u, v = (c1, c2) if u_first else (c2, c1)

    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = y
    if v_subsample == 2:
        u = u.repeat(2, axis=0)
        v = v.repeat(2, axis=0)
    yuv[:, :, 1] = u.repeat(2, axis=1)
    yuv[:, :, 2] = v.repeat(2, axis=1)

    return ycbcr_to_bgr888(yuv, options)


def y8_to_bgr888(
    data: npt.NDArray[np.uint8], w: int, h: int, stride: int, options: dict | None
) -> npt.NDArray[np.uint8]:
    color_range = options.get('range', 'full') if options else 'full'

    y = as_strided(data, shape=(h, w), strides=(stride, 1), writeable=False)

    if color_range == 'limited':
        # Convert from limited range (16-235) to full range (0-255)
        y = np.clip((y.astype(np.float32) - 16) * 255 / 219, 0, 255).astype(np.uint8)

    # Create grayscale RGB (Y becomes R=G=B)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = y  # B
    rgb[:, :, 1] = y  # G
    rgb[:, :, 2] = y  # R

    return rgb


def yuv_to_bgr888(
    arr: npt.NDArray[np.uint8],
    w: int,
    h: int,
    strides: tuple[int, ...],
    fmt: PixelFormat,
    options: dict | None,
) -> npt.NDArray[np.uint8]:

    # HACK: for backward compatibility. Drop when no external user calls this internal function.
    if isinstance(strides, int):
        strides = cast('tuple[int, ...]', (strides,))

    if fmt == PixelFormats.Y8:
        return y8_to_bgr888(arr, w, h, strides[0], options)

    if fmt == PixelFormats.YUYV:
        return yuyv_to_bgr888(arr, w, h, strides[0], options)

    if fmt == PixelFormats.UYVY:
        return uyvy_to_bgr888(arr, w, h, strides[0], options)

    if fmt == PixelFormats.NV12:
        return nv_to_bgr888(arr, w, h, strides[0], strides[1], 2, True, options)

    if fmt == PixelFormats.NV21:
        return nv_to_bgr888(arr, w, h, strides[0], strides[1], 2, False, options)

    if fmt == PixelFormats.NV16:
        return nv_to_bgr888(arr, w, h, strides[0], strides[1], 1, True, options)

    if fmt == PixelFormats.NV61:
        return nv_to_bgr888(arr, w, h, strides[0], strides[1], 1, False, options)

    if fmt == PixelFormats.YUV420:
        return planar_yuv_to_bgr888(arr, w, h, strides[0], strides[1], strides[2], 2, True, options)

    if fmt == PixelFormats.YVU420:
        return planar_yuv_to_bgr888(
            arr, w, h, strides[0], strides[1], strides[2], 2, False, options
        )

    if fmt == PixelFormats.YUV422:
        return planar_yuv_to_bgr888(arr, w, h, strides[0], strides[1], strides[2], 1, True, options)

    if fmt == PixelFormats.YVU422:
        return planar_yuv_to_bgr888(
            arr, w, h, strides[0], strides[1], strides[2], 1, False, options
        )

    raise RuntimeError(f'Unsupported YUV format {fmt}')
