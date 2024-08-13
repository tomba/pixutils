# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

import numpy as np

def rgb_to_rgb(fmt, w, h, data):
    if fmt == 'RGB888':
        rgb = data.reshape((h, w, 3))
        rgb = np.flip(rgb, axis=2) # Flip the components
    elif fmt == 'BGR888':
        rgb = data.reshape((h, w, 3))
    elif fmt in ['ARGB8888', 'XRGB8888']:
        rgb = data.reshape((h, w, 4))
        rgb = np.delete(rgb, np.s_[3::4], axis=2) # drop alpha component
        rgb = np.flip(rgb, axis=2) # Flip the components
    elif fmt in ['ABGR8888', 'XBGR8888']:
        rgb = data.reshape((h, w, 4))
        rgb = np.delete(rgb, np.s_[3::4], axis=2) # drop alpha component
    elif fmt == 'XBGR2101010':
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
        raise RuntimeError('Unsupported format ' + fmt)

    return rgb
