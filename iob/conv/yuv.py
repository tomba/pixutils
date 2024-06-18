# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

import numpy as np

# https://web.archive.org/web/20180423091842/http://www.equasys.de/colorconversion.html

def convert_ycbcr_bt601_limited_to_rgb(yuv):
    offset = np.array([-16, -128, -128])

    m = np.array([
             [ 1.164, 1.164, 1.164 ],
             [ 0, -0.392, 2.017 ],
             [ 1.596, -0.813, 0 ],
         ])

    rgb = np.dot(yuv + offset, m)

    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.astype(np.uint8)

    return rgb

def convert_ycbcr_bt601_full_to_rgb(yuv):
    offset = np.array([0, -128, -128])

    m = np.array([
             [ 1, 1, 1 ],
             [ 0, -0.343, 1.765 ],
             [ 1.4, -0.711, 0 ],
         ])

    rgb = np.dot(yuv + offset, m)

    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.astype(np.uint8)

    return rgb

# https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639

def convert_yuv444_to_rgb(yuv):
    m = np.array([
        [1.0, 1.0, 1.0],
        [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
        [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]
    ])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.astype(np.uint8)

    return rgb

def convert_yuyv(data, w, h):
    # YUV422
    yuyv = data.reshape((h, w // 2 * 4))

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = yuyv[:, 0::2]                    # Y
    yuv[:, :, 1] = yuyv[:, 1::4].repeat(2, axis=1)  # U
    yuv[:, :, 2] = yuyv[:, 3::4].repeat(2, axis=1)  # V

    return convert_ycbcr_bt601_full_to_rgb(yuv)


def convert_uyvy(data, w, h):
    # YUV422
    yuyv = data.reshape((h, w // 2 * 4))

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = yuyv[:, 1::2]                    # Y
    yuv[:, :, 1] = yuyv[:, 0::4].repeat(2, axis=1)  # U
    yuv[:, :, 2] = yuyv[:, 2::4].repeat(2, axis=1)  # V

    return convert_ycbcr_bt601_full_to_rgb(yuv)

def convert_nv12(data, w, h):
    plane1 = data[:w * h]
    plane2 = data[w * h:]

    y = plane1.reshape((h, w))
    uv = plane2.reshape((h // 2, w // 2, 2))

    # YUV444
    yuv = np.empty((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = y[:, :]                    # Y
    yuv[:, :, 1] = uv[:, :, 0].repeat(2, axis=0).repeat(2, axis=1)  # U
    yuv[:, :, 2] = uv[:, :, 1].repeat(2, axis=0).repeat(2, axis=1)  # V

    return convert_yuv444_to_rgb(yuv)

def convert_y8(data, w, h):
    y = data.reshape((h, w))

    # YUV444
    yuv = np.zeros((h, w, 3), dtype=np.uint8)
    yuv[:, :, 0] = y #yuyv[:, 0::2]                    # Y
    yuv[:, :, 1] = y  # U
    yuv[:, :, 2] = y  # V

    return yuv

    #return convert_yuv444_to_rgb(yuv)
