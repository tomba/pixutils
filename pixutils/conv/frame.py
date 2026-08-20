# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2026, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat

__all__ = ['Frame']


def _normalize_strides(
    fmt: PixelFormat, width: int, bytesperline: int | Sequence[int]
) -> tuple[int, ...]:
    """Normalize bytesperline to a per-plane tuple of concrete (non-zero) strides.

    ``bytesperline`` is either 0 (natural strides), a single non-zero int (stride
    of plane 0, others extrapolated), or one stride per plane.
    """
    n = len(fmt.planes)

    if isinstance(bytesperline, int):
        if bytesperline == 0:
            return tuple(fmt.stride(width, i) for i in range(n))
        return tuple(fmt.extrapolate_stride(bytesperline, i) for i in range(n))

    if len(bytesperline) != n:
        raise ValueError(
            f'Strides sequence length {len(bytesperline)} does not match number of planes {n}'
        )
    if any(s == 0 for s in bytesperline):
        raise ValueError('Strides sequence must contain non-zero stride for each plane')
    return tuple(bytesperline)


@dataclass(frozen=True, eq=False)
class Frame:
    """A (possibly multi-plane) framebuffer: pixel format, dimensions, and one
    independent buffer + row stride per plane.

    Each plane is a standalone 1-D ``uint8`` array whose byte 0 is that plane's
    top-left pixel, with its own row pitch in ``strides``. Planes may be
    independent allocations or views into a shared buffer.

    ``frozen`` makes the attributes read-only, but the planes it points at are
    mutable ndarrays, so a Frame has reference rather than value semantics:
    ``eq=False`` keeps the identity-based ``__eq__``/``__hash__`` inherited from
    ``object`` instead of the field-wise ones a dataclass would generate (which
    would raise on the ndarray fields).

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
        strides = _normalize_strides(fmt, width, bytesperline)

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

    @classmethod
    def from_planes(
        cls,
        fmt: PixelFormat,
        width: int,
        height: int,
        planes: Sequence[npt.NDArray[np.uint8]],
        bytesperline: int | Sequence[int] = 0,
    ) -> Frame:
        """Build a Frame from one independent buffer per plane.

        ``planes`` holds one array per plane (``len(fmt.planes)`` of them); each is
        flattened to a 1-D ``uint8`` view and sliced to its plane size.
        ``bytesperline`` follows the same convention as :meth:`from_single_buffer`.
        """
        n = len(fmt.planes)
        if len(planes) != n:
            raise ValueError(
                f'Number of planes {len(planes)} does not match format {fmt.name} ({n})'
            )

        strides = _normalize_strides(fmt, width, bytesperline)

        plane_arrs = []
        for i in range(n):
            arr = np.ascontiguousarray(planes[i]).reshape(-1).view(np.uint8)
            if strides[i] < fmt.stride(width, i):
                raise ValueError('bytesperline is too small')
            plane_size = fmt.planesize(strides[i], height, i)
            if arr.size < plane_size:
                raise ValueError(f'Plane {i} is too small: {arr.size} < {plane_size}')
            plane_arrs.append(arr[:plane_size])

        return cls(fmt, width, height, tuple(plane_arrs), strides)

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

    def crop_align(self) -> tuple[int, int]:
        """Return the ``(x, y)`` pixel alignment that :meth:`crop` requires.

        This is ``fmt.pixel_align`` widened by what the plane layout needs: a
        crop origin has to land on a whole macropixel block of every plane, and
        on a whole chroma sample of every subsampled plane. Most formats
        declare that in ``pixel_align`` already, but a few do not --
        Y210/Y212/Y216 pack two pixels into one block while declaring
        ``pixel_align (1, 1)``, and YUV420/YUV422 (plus their VU twins) declare
        ``(1, 1)`` with 2x-subsampled chroma planes.
        """
        fmt = self.fmt
        ax, ay = fmt.pixel_align
        for pi in fmt.planes:
            ax = math.lcm(ax, pi.pixels_per_block * pi.hsub)
            ay = math.lcm(ay, pi.vsub)
        return (ax, ay)

    def crop(self, x: int, y: int, w: int, h: int) -> Frame:
        """Return a new Frame for the sub-region ``(x, y, w, h)``.

        Each plane view is offset to the crop origin and keeps its original
        stride; the new Frame's width/height are the crop size. ``x``/``w`` must
        be multiples of ``crop_align()[0]`` and ``y``/``h`` of
        ``crop_align()[1]``, and the region must lie within the frame;
        otherwise ``ValueError`` is raised.

        Each plane view ends at the last byte the crop covers, so the plane
        sizes stay consistent with the crop size and :meth:`combined` keeps
        working.
        """
        fmt = self.fmt
        ax, ay = self.crop_align()

        if x % ax or w % ax or y % ay or h % ay:
            raise ValueError(
                f'Crop ({x}, {y}, {w}, {h}) is not aligned to {fmt.name} '
                f'crop alignment ({ax}, {ay})'
            )
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > self.width or y + h > self.height:
            raise ValueError(
                f'Crop ({x}, {y}, {w}, {h}) is out of bounds for {self.width}x{self.height}'
            )

        new_planes = []
        for i, pi in enumerate(fmt.planes):
            stride = self.strides[i]
            x_bytes = (x // pi.pixels_per_block) * pi.bytes_per_block // pi.hsub
            start = (y // pi.vsub) * stride + x_bytes
            # Full rows for all but the last one, which is only as wide as the
            # crop. Trimming here keeps the plane in step with the crop size,
            # so a full-width crop is a plain frame again and combined() stays
            # meaningful.
            size = (h // pi.vsub - 1) * stride + fmt.stride(w, i)
            new_planes.append(self.planes[i][start : start + size])

        return Frame(fmt, w, h, tuple(new_planes), self.strides)
