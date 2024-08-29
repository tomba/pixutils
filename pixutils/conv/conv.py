# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pixutils import PixelFormat, PixelFormats

from .yuv import y8_to_bgr888, yuyv_to_bgr888, uyvy_to_bgr888, nv12_to_bgr888
from .rgb import rgb_to_bgr888
from .raw import raw_to_bgr888

def to_bgr888(fmt: PixelFormat, w, h, bytesperline, arr: npt.NDArray[np.uint8],
              options: None | dict = None):
    if fmt == PixelFormats.Y8:
        return y8_to_bgr888(arr, w, h)

    if fmt == PixelFormats.YUYV:
        return yuyv_to_bgr888(arr, w, h, options)

    if fmt == PixelFormats.UYVY:
        return uyvy_to_bgr888(arr, w, h, options)

    if fmt == PixelFormats.NV12:
        return nv12_to_bgr888(arr, w, h, options)

    if fmt.name.startswith('S'):
        return raw_to_bgr888(arr, w, h, bytesperline, fmt)

    if 'RGB' in fmt.name or 'BGR' in fmt.name:
        return rgb_to_bgr888(fmt, w, h, arr)

    raise RuntimeError(f'Unsupported format {fmt}')

def buffer_to_bgr888(fmt: PixelFormat, w, h, bytesperline, buffer,
                     options: None | dict = None):
    arr = np.frombuffer(buffer, dtype=np.uint8)
    rgb = to_bgr888(fmt, w, h, bytesperline, arr, options)
    return rgb
