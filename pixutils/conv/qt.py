# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

from __future__ import annotations

import numpy.typing as npt
import numpy as np
from PyQt6 import QtGui
from pixutils.formats import PixelFormat, PixelFormats
from .conv import buffer_to_bgr888


def bgr888_to_pix(rgb: npt.NDArray[np.uint8]) -> QtGui.QPixmap:
    # QImage doesn't seem to like a numpy view
    if rgb.base is not None:
        rgb = rgb.copy()

    w = rgb.shape[1]
    h = rgb.shape[0]
    qim = QtGui.QImage(rgb, w, h, QtGui.QImage.Format.Format_RGB888) # pylint: disable=no-member # type: ignore
    pix = QtGui.QPixmap.fromImage(qim)
    return pix


def buffer_to_pix(fmt: PixelFormat, w: int, h: int, bytesperline: int, buffer, options: None | dict = None) -> QtGui.QPixmap:
    if fmt == PixelFormats.MJPEG:
        pix = QtGui.QPixmap(w, h)
        pix.loadFromData(buffer)
    else:
        rgb = buffer_to_bgr888(fmt, w, h, bytesperline, buffer, options)
        pix = bgr888_to_pix(rgb)

    return pix
