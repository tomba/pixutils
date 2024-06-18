# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

import numpy as np

from .yuv import convert_y8, convert_yuyv, convert_uyvy, convert_nv12
from .rgb import rgb_to_rgb
from .raw import convert_raw

def to_rgb(fmt, w, h, bytesperline, data):
    if fmt == 'Y8':
        return convert_y8(data, w, h)

    if fmt == 'YUYV':
        return convert_yuyv(data, w, h)

    if fmt == 'UYVY':
        return convert_uyvy(data, w, h)

    if fmt == 'NV12':
        return convert_nv12(data, w, h)

    if fmt.startswith('S'):
        return convert_raw(data, w, h, bytesperline, fmt)

    if 'RGB' in fmt or 'BGR' in fmt:
        return rgb_to_rgb(fmt, w, h, data)

    raise RuntimeError('Unsupported format ' + fmt)

def data_to_rgb(fmt, w, h, bytesperline, data):
    data = np.frombuffer(data, dtype=np.uint8)
    rgb = to_rgb(fmt, w, h, bytesperline, data)
    return rgb
