# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat

__all__ = ['Frame']


@dataclass(frozen=True)
class Frame:
    """A (possibly multi-plane) framebuffer: pixel format, dimensions, and one
    independent buffer + row stride per plane.

    Each plane is a standalone 1-D ``uint8`` array whose byte 0 is that plane's
    top-left pixel, with its own row pitch in ``strides``. Planes may be
    independent allocations or views into a shared buffer.

    Attributes:
        fmt: The pixel format.
        width: Image width in pixels.
        height: Image height in pixels.
        planes: One 1-D ``uint8`` array per plane (``len(fmt.planes)`` of them).
        strides: Per-plane row stride in bytes (same length as ``planes``).
    """

    fmt: PixelFormat
    width: int
    height: int
    planes: tuple[npt.NDArray[np.uint8], ...]
    strides: tuple[int, ...]

    # Zero-copy backing for combined() when the planes were carved from one
    # buffer. Transitional: backends that still consume a single concatenated
    # blob use combined(); not part of identity.
    _backing: npt.NDArray[np.uint8] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_single_buffer(
        cls,
        fmt: PixelFormat,
        width: int,
        height: int,
        bytesperline: int | Sequence[int],
        arr: npt.NDArray[np.uint8],
    ) -> Frame:
        """Build a Frame from one buffer holding all planes back-to-back.

        ``bytesperline`` follows the same convention as
        :func:`pixutils.conv.to_bgr888`:

        - ``0``: natural strides for every plane,
        - a single non-zero int: stride of plane 0, other planes extrapolated,
        - a sequence of non-zero ints: one stride per plane.
        """
        arr = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)

        n = len(fmt.planes)

        # Normalize bytesperline to a per-plane tuple of concrete (non-zero) strides
        if isinstance(bytesperline, int):
            if bytesperline == 0:
                strides = tuple(fmt.stride(width, i) for i in range(n))
            else:
                strides = tuple(fmt.extrapolate_stride(bytesperline, i) for i in range(n))
        else:
            if len(bytesperline) != n:
                raise ValueError(
                    f'Strides sequence length {len(bytesperline)} does not match number of planes {n}'
                )
            if any(s == 0 for s in bytesperline):
                raise ValueError('Strides sequence must contain non-zero stride for each plane')
            strides = tuple(bytesperline)

        sizes = []
        total = 0
        for i in range(n):
            if strides[i] < fmt.stride(width, i):
                raise ValueError('bytesperline is too small')

            plane_size = fmt.planesize(strides[i], height, i)
            if arr.size < plane_size:
                raise ValueError(
                    f'Input array is too small: {arr.size} < {plane_size}, {bytesperline}, {strides}'
                )

            sizes.append(plane_size)
            total += plane_size

        arr = arr[:total]

        planes = []
        offset = 0
        for plane_size in sizes:
            planes.append(arr[offset : offset + plane_size])
            offset += plane_size

        return cls(fmt, width, height, tuple(planes), strides, _backing=arr)

    def combined(self) -> npt.NDArray[np.uint8]:
        """Return all planes as one contiguous 1-D ``uint8`` buffer.

        Zero-copy when the planes were carved from a single buffer (the case
        produced by :meth:`from_single_buffer`); copies otherwise. Transitional
        helper for backends that still consume a single concatenated blob.
        """
        if self._backing is not None:
            return self._backing
        if len(self.planes) == 1:
            return self.planes[0]
        return np.concatenate(self.planes)
