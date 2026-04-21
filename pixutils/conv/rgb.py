# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided

from pixutils.formats import PixelFormat, PixelFormats


def rgb_to_bgr888(
    fmt: PixelFormat, w: int, h: int, strides: tuple[int, ...], data: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    stride = strides[0]

    if fmt == PixelFormats.RGB888:
        rgb = as_strided(data, shape=(h, w, 3), strides=(stride, 3, 1), writeable=False)
        rgb = np.flip(rgb, axis=2)
    elif fmt == PixelFormats.BGR888:
        rgb = as_strided(data, shape=(h, w, 3), strides=(stride, 3, 1), writeable=False)
        rgb = rgb.copy()
    elif fmt in [PixelFormats.ARGB8888, PixelFormats.XRGB8888]:
        rgb = as_strided(data, shape=(h, w, 4), strides=(stride, 4, 1), writeable=False)
        rgb = np.delete(rgb, np.s_[3::4], axis=2)  # drop alpha component
        rgb = np.flip(rgb, axis=2)
    elif fmt in [PixelFormats.ABGR8888, PixelFormats.XBGR8888]:
        rgb = as_strided(data, shape=(h, w, 4), strides=(stride, 4, 1), writeable=False)
        rgb = np.delete(rgb, np.s_[3::4], axis=2)  # drop alpha component
    elif fmt == PixelFormats.XBGR2101010:
        v = as_strided(
            data.view(np.dtype('<u4')), shape=(h, w), strides=(stride, 4), writeable=False
        )

        output = np.zeros((h, w, 3), dtype=np.uint16)

        output[:, :, 0] = v & 0x3FF  # R
        output[:, :, 1] = (v >> 10) & 0x3FF  # G
        output[:, :, 2] = (v >> 20) & 0x3FF  # B

        rgb = output

        rgb >>= 10 - 8
        rgb = rgb.astype(np.uint8)

        # rgb = np.delete(rgb, np.s_[3::4], axis=2) # drop alpha component
        # rgb = np.flip(rgb, axis=2) # Flip the components
    else:
        raise RuntimeError(f'Unsupported RGB format {fmt}')

    return rgb
