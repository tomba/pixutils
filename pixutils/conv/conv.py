# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat, PixelColorEncoding

from .yuv import yuv_to_bgr888
from .rgb import rgb_to_bgr888
from .raw import raw_to_bgr888

def to_bgr888(fmt: PixelFormat, w, h, bytesperline, arr: npt.NDArray[np.uint8],
              options: None | dict = None):
    if fmt.color == PixelColorEncoding.YUV:
        return yuv_to_bgr888(arr, w, h, fmt, options)

    if fmt.color == PixelColorEncoding.RAW:
        return raw_to_bgr888(arr, w, h, bytesperline, fmt)

    if fmt.color == PixelColorEncoding.RGB:
        return rgb_to_bgr888(fmt, w, h, arr)

    raise RuntimeError(f'Unsupported format {fmt}')

def buffer_to_bgr888(fmt: PixelFormat, w, h, bytesperline, buffer,
                     options: None | dict = None):
    arr = np.frombuffer(buffer, dtype=np.uint8)
    rgb = to_bgr888(fmt, w, h, bytesperline, arr, options)
    return rgb
