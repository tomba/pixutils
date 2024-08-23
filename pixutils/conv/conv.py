# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np

from pixutils import PixelFormat, PixelFormats

from .yuv import convert_y8, convert_yuyv, convert_uyvy, convert_nv12
from .rgb import rgb_to_rgb
from .raw import convert_raw

def to_rgb(fmt: PixelFormat, w, h, bytesperline, data,
           options: None | dict = None):
    if fmt == PixelFormats.Y8:
        return convert_y8(data, w, h)

    if fmt == PixelFormats.YUYV:
        return convert_yuyv(data, w, h, options)

    if fmt == PixelFormats.UYVY:
        return convert_uyvy(data, w, h, options)

    if fmt == PixelFormats.NV12:
        return convert_nv12(data, w, h, options)

    if fmt.name.startswith('S'):
        return convert_raw(data, w, h, bytesperline, fmt)

    if 'RGB' in fmt.name or 'BGR' in fmt.name:
        return rgb_to_rgb(fmt, w, h, data)

    raise RuntimeError(f'Unsupported format {fmt}')

def data_to_rgb(fmt: PixelFormat, w, h, bytesperline, data,
                options: None | dict = None):
    data = np.frombuffer(data, dtype=np.uint8)
    rgb = to_rgb(fmt, w, h, bytesperline, data, options)
    return rgb
