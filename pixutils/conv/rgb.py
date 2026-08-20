# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided

from pixutils.formats import PixelFormats

from .frame import Frame


def rgb_to_bgr888(frame: Frame, options: dict | None) -> npt.NDArray[np.uint8] | None:
    fmt = frame.fmt
    w = frame.width
    h = frame.height
    stride = frame.strides[0]
    data = frame.planes[0]

    if fmt == PixelFormats.RGB888:
        src = as_strided(data, shape=(h, w, 3), strides=(stride, 3, 1), writeable=False)
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = src[..., 2]
        rgb[..., 1] = src[..., 1]
        rgb[..., 2] = src[..., 0]
    elif fmt == PixelFormats.BGR888:
        rgb = as_strided(data, shape=(h, w, 3), strides=(stride, 3, 1), writeable=False)
    elif fmt in [PixelFormats.ARGB8888, PixelFormats.XRGB8888]:
        src = as_strided(data, shape=(h, w, 4), strides=(stride, 4, 1), writeable=False)
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = src[..., 2]
        rgb[..., 1] = src[..., 1]
        rgb[..., 2] = src[..., 0]
    elif fmt in [PixelFormats.ABGR8888, PixelFormats.XBGR8888]:
        src = as_strided(data, shape=(h, w, 4), strides=(stride, 4, 1), writeable=False)
        rgb = src[..., :3]
    elif fmt == PixelFormats.XBGR2101010:
        v = as_strided(
            data.view(np.dtype('<u4')), shape=(h, w), strides=(stride, 4), writeable=False
        )

        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (v >> 2) & 0xFF  # R (10-bit → 8-bit)
        rgb[:, :, 1] = (v >> 12) & 0xFF  # G
        rgb[:, :, 2] = (v >> 22) & 0xFF  # B
    else:
        return None

    return rgb
