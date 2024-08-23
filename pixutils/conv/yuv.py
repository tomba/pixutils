# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

import numpy as np

YCBCR_VALUES = {
    'bt601': {
        'limited': {
            'offsets': (-16, -128, -128),
            'matrix': [
                [1.1643854428931106, 1.1643854428931106, 1.1643854428931106],
                [0.0, -0.3917753871976806, 2.01722743572137],
                [1.596032654306859, -0.8129854276340887, 0.0]
            ],
        },
        'full': {
            'offsets': (0, -128, -128),
            'matrix': [
                [1.0, 1.0, 1.0],
                [0.0, -0.3441367208361944, 1.7720149538414587],
                [1.4019989318684671, -0.714152742809186, 0.0]
            ],
        },
    },
}


def convert_ycbcr(yuv, options):
    color_range = 'limited'
    color_encoding = 'bt601'

    if options:
        color_range = options.get('range', color_range)
        color_encoding = options.get('encoding', color_encoding)

    conv_data = YCBCR_VALUES[color_encoding][color_range]

    offset = np.array(conv_data['offsets'])

    m = np.array(conv_data['matrix'])

    rgb = np.dot(yuv + offset, m)

    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.astype(np.uint8)

    return rgb


def convert_yuyv(data, w, h, options):
    # YUV422
    yuyv = data.reshape((h, w // 2 * 4))

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = yuyv[:, 0::2]                    # Y
    yuv[:, :, 1] = yuyv[:, 1::4].repeat(2, axis=1)  # U
    yuv[:, :, 2] = yuyv[:, 3::4].repeat(2, axis=1)  # V

    return convert_ycbcr(yuv, options)


def convert_uyvy(data, w, h, options):
    # YUV422
    yuyv = data.reshape((h, w // 2 * 4))

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = yuyv[:, 1::2]                    # Y
    yuv[:, :, 1] = yuyv[:, 0::4].repeat(2, axis=1)  # U
    yuv[:, :, 2] = yuyv[:, 2::4].repeat(2, axis=1)  # V

    return convert_ycbcr(yuv, options)

def convert_nv12(data, w, h, options):
    plane1 = data[:w * h]
    plane2 = data[w * h:]

    y = plane1.reshape((h, w))
    uv = plane2.reshape((h // 2, w // 2, 2))

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = y[:, :]                    # Y
    yuv[:, :, 1] = uv[:, :, 0].repeat(2, axis=0).repeat(2, axis=1)  # U
    yuv[:, :, 2] = uv[:, :, 1].repeat(2, axis=0).repeat(2, axis=1)  # V

    return convert_ycbcr(yuv, options)

def convert_y8(data, w, h):
    y = data.reshape((h, w))

    # YUV444
    yuv = np.zeros((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = y #yuyv[:, 0::2]                    # Y
    yuv[:, :, 1] = y  # U
    yuv[:, :, 2] = y  # V

    return yuv
