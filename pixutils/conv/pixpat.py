# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pixpat as _pixpat

from pixutils.formats import PixelFormat

from .frame import Frame

__all__ = ['pixpat_to_bgr888']

_REC_MAP = {
    'bt601': _pixpat.Rec.BT601,
    'bt709': _pixpat.Rec.BT709,
    'bt2020': _pixpat.Rec.BT2020,
}

_RANGE_MAP = {
    'limited': _pixpat.Range.LIMITED,
    'full': _pixpat.Range.FULL,
}


def _can_use_pixpat(fmt: PixelFormat, options: dict | None) -> bool:
    if not _pixpat.is_supported(fmt.name):
        return False

    if options:
        if options.get('encoding', 'bt601') not in _REC_MAP:
            return False
        if options.get('range', 'limited') not in _RANGE_MAP:
            return False

    return True


def pixpat_to_bgr888(frame: Frame, options: dict | None) -> npt.NDArray[np.uint8] | None:
    """Try to convert using pixpat. Returns None if pixpat can't handle this format."""
    fmt = frame.fmt

    if not _can_use_pixpat(fmt, options):
        return None

    encoding = (options or {}).get('encoding', 'bt601')
    color_range = (options or {}).get('range', 'limited')

    src = _pixpat.Buffer(
        planes=list(frame.planes),
        fmt=fmt.name,
        width=frame.width,
        height=frame.height,
        strides=list(frame.strides),
    )

    dst_arr = np.empty((frame.height, frame.width, 3), dtype=np.uint8)
    dst = _pixpat.Buffer(
        planes=[dst_arr],
        fmt='BGR888',
        width=frame.width,
        height=frame.height,
        strides=[frame.width * 3],
    )

    try:
        _pixpat.convert(
            dst,
            src,
            rec=_REC_MAP[encoding],
            color_range=_RANGE_MAP[color_range],
        )
    except _pixpat.PixpatError:
        return None

    return dst_arr
