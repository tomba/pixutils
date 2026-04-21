# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat


# Note: the callers of this function should be fixed to handle
# the stride properly
def strip_padding(
    data: npt.NDArray[np.uint8], height: int, strides: tuple[int, ...], fmt: PixelFormat, width: int
) -> npt.NDArray[np.uint8]:
    if all(strides[i] == fmt.stride(width, i) for i in range(len(fmt.planes))):
        return data

    planes = []
    offset = 0
    for i, plane in enumerate(fmt.planes):
        plane_h = height // plane.vsub
        row_bytes = fmt.stride(width, i)
        plane_data = data[offset : offset + strides[i] * plane_h]
        if strides[i] != row_bytes:
            plane_data = plane_data.reshape((plane_h, strides[i]))[:, :row_bytes].flatten()
        planes.append(plane_data)
        offset += strides[i] * plane_h
    return np.concatenate(planes) if len(planes) > 1 else planes[0]
