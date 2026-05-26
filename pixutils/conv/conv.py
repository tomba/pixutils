# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelColorEncoding, PixelFormat

from .backends import get_backends
from .frame import Frame
from .raw import raw_to_bgr888
from .rgb import rgb_to_bgr888
from .yuv import yuv_to_bgr888


def to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int | Sequence[int],
    arr: npt.NDArray[np.uint8] | Sequence[npt.NDArray[np.uint8]],
    options: None | dict = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Convert a numpy array containing pixel data to an 8-bit 3-channel image.

    Input convention
    ----------------
    Input pixel formats follow the Linux DRM/V4L2 fourcc naming convention,
    where the name describes a little-endian machine word from MSB to LSB.
    The in-memory byte order is therefore the reverse of the name. Examples
    (each byte listed starting at byte 0 of a pixel):

    - ``RGB888``   -> B, G, R
    - ``BGR888``   -> R, G, B
    - ``XRGB8888`` -> B, G, R, X
    - ``BGRX8888`` -> X, R, G, B
    - ``ARGB8888`` -> B, G, R, A
    - ``ABGR8888`` -> R, G, B, A

    Output layout
    -------------
    The return value is a ``(height, width, 3)`` uint8 array. For each pixel,
    byte 0 is R, byte 1 is G, byte 2 is B. In DRM fourcc terms this is
    ``BGR888`` (hence the function name), so the output memory layout follows
    the same DRM/V4L2 fourcc convention as the inputs documented above.

    Other frameworks label this same ``(R, G, B)``-in-memory layout
    differently, typically because they name formats by in-memory byte order
    rather than by little-endian machine word: Qt calls it
    ``QImage::Format_RGB888``, PIL calls it the ``'RGB'`` mode, V4L2 (in its
    userspace API) calls it ``V4L2_PIX_FMT_RGB24``, and OpenCV calls it
    ``RGB``. The result can be passed to any of these without channel
    reordering. OpenCV's default entry points (``cv2.imread``,
    ``cv2.imshow``, ``cv2.imwrite``) instead expect OpenCV-``BGR`` — bytes
    ``(B, G, R)`` in memory — and therefore require a channel swap.

    Parameters:
        fmt: The pixel format of the input data
        width: Width of the image in pixels
        height: Height of the image in pixels
        bytesperline: Bytes per line. Either:
            - 0: no padding, natural strides are used for every plane
            - a single non-zero int: stride of plane 0. For multiplane formats the
              strides of the other planes are extrapolated, preserving the padding
              ratio (matches libcamera convention)
            - a sequence of non-zero ints: one stride per plane
        arr: Either a single numpy array holding all planes back-to-back, or a
            sequence of per-plane numpy arrays (one per plane of ``fmt``)
        options: Optional dictionary with conversion options:
            - backends: List of backends in priority order, e.g. ['opencv', 'numba']
            - range: 'limited' or 'full' (for YUV formats)
            - encoding: 'bt601' (for YUV formats)
            - demosaic_method: '3x3', 'bilinear', 'mosaic', or 'opencv' (for RAW formats)
        crop: Optional ``(x, y, w, h)`` sub-region to convert. ``width``/``height``
            still describe the full source buffer; ``crop`` selects the output
            region. See :meth:`Frame.crop` for alignment requirements.

    Returns:
        ``(height, width, 3)`` uint8 array with per-pixel bytes (R, G, B)
        (see "Output layout" above).
    """

    if isinstance(arr, np.ndarray):
        frame = Frame.from_single_buffer(fmt, width, height, bytesperline, arr)
    else:
        frame = Frame.from_planes(fmt, width, height, arr, bytesperline)
    return frame_to_bgr888(frame, options, crop=crop)


def frame_to_bgr888(
    frame: Frame,
    options: None | dict = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Convert a :class:`Frame` to an 8-bit 3-channel image.

    See :func:`to_bgr888` for the input format conventions and the output byte
    layout.

    Parameters:
        frame: The framebuffer to convert (pixel format, dimensions, and one
            buffer + stride per plane).
        options: Optional dictionary with conversion options (see
            :func:`to_bgr888`).
        crop: Optional ``(x, y, w, h)`` sub-region to convert; see
            :meth:`Frame.crop` for alignment requirements.

    Returns:
        See :func:`to_bgr888`.
    """

    if crop is not None:
        frame = frame.crop(*crop)

    fmt = frame.fmt

    # Get list of backends to try
    backends = get_backends(options.get('backends') if options else None)
    if not backends:
        raise NotImplementedError('No backends available')

    # Try backends in priority order
    result = None
    for backend in backends:
        if backend == 'opencv':
            from .opencv import opencv_to_bgr888

            result = opencv_to_bgr888(frame, options)
            if result is not None:
                break
            # opencv couldn't handle this format/options, try next backend
            continue
        elif backend == 'pixpat':
            from .pixpat import pixpat_to_bgr888

            result = pixpat_to_bgr888(frame, options)
            if result is not None:
                break
            # pixpat couldn't handle this format/options, try next backend
            continue
        elif backend == 'numba':
            from .numba import numba_to_bgr888

            result = numba_to_bgr888(frame, options)
            if result is not None:
                break
        elif backend == 'numpy':
            if fmt.color == PixelColorEncoding.YUV:
                result = yuv_to_bgr888(frame, options)
            elif fmt.color == PixelColorEncoding.RAW:
                result = raw_to_bgr888(frame, options)
            elif fmt.color == PixelColorEncoding.RGB:
                result = rgb_to_bgr888(frame, options)
            else:
                raise NotImplementedError(f'Unsupported format {fmt}')
            break

    if result is None:
        raise NotImplementedError(f'No backend could handle {fmt.name} with given options')

    # Backends may return a view; guarantee only that it doesn't alias the
    # input buffer. Callers that need a specific layout can contiguify
    # themselves.
    if np.shares_memory(result, frame.combined()):
        result = result.copy()
    return result


def buffer_to_bgr888(
    fmt: PixelFormat,
    width: int,
    height: int,
    bytesperline: int | Sequence[int],
    buffer,
    options: None | dict = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Convert a buffer-like object containing pixel data to an 8-bit 3-channel
    image. Thin wrapper around :func:`to_bgr888`; see that function for input
    format conventions and output byte layout.

    Parameters:
        fmt: The pixel format of the input data
        width: Width of the image in pixels
        height: Height of the image in pixels
        bytesperline: Bytes per line. Either 0 (natural strides), a single
            non-zero int (stride of plane 0; other planes extrapolated), or a
            sequence of non-zero ints with one value per plane
        buffer: A single buffer-like object holding all planes, or a list/tuple
            of per-plane buffer-like objects (one per plane of ``fmt``)
        options: Optional dictionary with conversion options

    Returns:
        See :func:`to_bgr888`.

    TODO:
        3.12+ supports collections.abc.Buffer which could be used for the input
        buffer
    """

    if isinstance(buffer, (list, tuple)):
        arr: npt.NDArray | list = [np.frombuffer(b, dtype=np.uint8) for b in buffer]
    else:
        arr = np.frombuffer(buffer, dtype=np.uint8)
    return to_bgr888(fmt, width, height, bytesperline, arr, options, crop=crop)
