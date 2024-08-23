# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

from PyQt6 import QtGui
from pixutils import PixelFormat
from .conv import PixelFormats, data_to_rgb

def rgb_to_pix(rgb):
    # QImage doesn't seem to like a numpy view
    if rgb.base is not None:
        rgb = rgb.copy()

    w = rgb.shape[1]
    h = rgb.shape[0]
    qim = QtGui.QImage(rgb, w, h, QtGui.QImage.Format.Format_RGB888) # pylint: disable=no-member
    pix = QtGui.QPixmap.fromImage(qim)
    return pix


def data_to_pix(fmt: PixelFormat, w, h, bytesperline, data):
    print("SRC:", list(data[0:4]))

    if fmt == PixelFormats.MJPEG:
        pix = QtGui.QPixmap(w, h)
        pix.loadFromData(data)
    else:
        rgb = data_to_rgb(fmt, w, h, bytesperline, data)

        print("RGB:", list(rgb[0,0]))
        #print(rgb)

        pix = rgb_to_pix(rgb)

    return pix
