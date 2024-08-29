# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

import numpy as np
import numpy.typing as npt

from pixutils import PixelFormat, PixelFormats

def rgb_to_bgr888(fmt: PixelFormat, w, h, data: npt.NDArray[np.uint8]):
    if fmt == PixelFormats.RGB888:
        rgb = data.reshape((h, w, 3))
        rgb = np.flip(rgb, axis=2) # Flip the components
    elif fmt == PixelFormats.BGR888:
        rgb = data.reshape((h, w, 3))
    elif fmt in [PixelFormats.ARGB8888, PixelFormats.XRGB8888]:
        rgb = data.reshape((h, w, 4))
        rgb = np.delete(rgb, np.s_[3::4], axis=2) # drop alpha component
        rgb = np.flip(rgb, axis=2) # Flip the components
    elif fmt in [PixelFormats.ABGR8888, PixelFormats.XBGR8888]:
        rgb = data.reshape((h, w, 4))
        rgb = np.delete(rgb, np.s_[3::4], axis=2) # drop alpha component
    elif fmt == PixelFormats.XBGR2101010:
        rgb = data.reshape((h, w * 4)) #.astype(np.uint16)

        print(rgb.shape, rgb.dtype)

        v = rgb.view(np.dtype('<u4'))

        print('{:#x}'.format(v[0, 0]))

        output = np.zeros((h, w, 3), dtype=np.uint16)

        output[:, :, 0] = v & 0x3ff             # R
        output[:, :, 1] = (v >> 10) & 0x3ff     # G
        output[:, :, 2] = (v >> 20) & 0x3ff     # B

        rgb = output

        print('{}'.format(rgb[0, 0]))

        rgb >>= 10 - 8
        rgb = rgb.astype(np.uint8)

        #rgb = np.delete(rgb, np.s_[3::4], axis=2) # drop alpha component
        #rgb = np.flip(rgb, axis=2) # Flip the components
    else:
        raise RuntimeError(f'Unsupported RGB format {fmt}')

    return rgb
