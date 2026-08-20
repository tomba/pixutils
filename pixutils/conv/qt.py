# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PyQt6 import QtGui

from pixutils.formats import PixelFormat, PixelFormats

from .conv import buffer_to_bgr888


def bgr888_to_pix(rgb: npt.NDArray[np.uint8]) -> QtGui.QPixmap:
    # Make sure we provide a contiguous array to QImage
    rgb = np.ascontiguousarray(rgb)

    w = rgb.shape[1]
    h = rgb.shape[0]
    qim = QtGui.QImage(rgb, w, h, QtGui.QImage.Format.Format_RGB888)  # type: ignore
    pix = QtGui.QPixmap.fromImage(qim)
    return pix


def buffer_to_pix(
    fmt: PixelFormat, w: int, h: int, bytesperline: int, buffer, options: None | dict = None
) -> QtGui.QPixmap:
    if fmt == PixelFormats.MJPEG:
        pix = QtGui.QPixmap(w, h)
        pix.loadFromData(buffer)
    else:
        rgb = buffer_to_bgr888(fmt, w, h, bytesperline, buffer, options)
        pix = bgr888_to_pix(rgb)

    return pix
