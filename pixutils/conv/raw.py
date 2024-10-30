# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from numpy.lib.stride_tricks import as_strided
import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat

__all__ = ['raw_to_bgr888']

# Debayering code from PiCamera documentation


def demosaic(data: npt.NDArray[np.uint16], r0, g0, g1, b0):
    # Separate the components from the Bayer data to RGB planes

    rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[r0[1] :: 2, r0[0] :: 2]  # Red
    rgb[0::2, 0::2, 1] = data[g0[1] :: 2, g0[0] :: 2]  # Green
    rgb[1::2, 1::2, 1] = data[g1[1] :: 2, g1[0] :: 2]  # Green
    rgb[0::2, 1::2, 2] = data[b0[1] :: 2, b0[0] :: 2]  # Blue

    # Below we present a fairly naive de-mosaic method that simply
    # calculates the weighted average of a pixel based on the pixels
    # surrounding it. The weighting is provided by a byte representation of
    # the Bayer filter which we construct first:

    bayer = np.zeros(rgb.shape, dtype=np.uint8)
    bayer[1::2, 0::2, 0] = 1  # Red
    bayer[0::2, 0::2, 1] = 1  # Green
    bayer[1::2, 1::2, 1] = 1  # Green
    bayer[0::2, 1::2, 2] = 1  # Blue

    # Allocate an array to hold our output with the same shape as the input
    # data. After this we define the size of window that will be used to
    # calculate each weighted average (3x3). Then we pad out the rgb and
    # bayer arrays, adding blank pixels at their edges to compensate for the
    # size of the window when calculating averages for edge pixels.

    output = np.empty(rgb.shape, dtype=rgb.dtype)
    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)

    rgb = np.pad(
        rgb,
        [
            (border[0], border[0]),
            (border[1], border[1]),
            (0, 0),
        ],
        'constant',
    )
    bayer = np.pad(
        bayer,
        [
            (border[0], border[0]),
            (border[1], border[1]),
            (0, 0),
        ],
        'constant',
    )

    # For each plane in the RGB data, we use a nifty numpy trick
    # (as_strided) to construct a view over the plane of 3x3 matrices. We do
    # the same for the bayer array, then use Einstein summation on each
    # (np.sum is simpler, but copies the data so it's slower), and divide
    # the results to get our weighted average:

    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]

        pview = as_strided(
            p,
            shape=(p.shape[0] - borders[0], p.shape[1] - borders[1]) + window,
            strides=p.strides * 2,
        )
        bview = as_strided(
            b,
            shape=(b.shape[0] - borders[0], b.shape[1] - borders[1]) + window,
            strides=b.strides * 2,
        )
        psum = np.einsum('ijkl->ij', pview)
        bsum = np.einsum('ijkl->ij', bview)
        output[..., plane] = psum // bsum

    return output


def raw_packed_to_bgr888(data: npt.NDArray[np.uint8], w, h, bytesperline, fmt: PixelFormat):
    fmtname = fmt.name

    bayer_pattern = fmtname[1:5]

    packed = fmtname.endswith('P')

    if packed:
        bitspp = int(fmtname[5:-1])
    else:
        bitspp = int(fmtname[5:])

    if bytesperline:
        data = data.reshape((len(data) // bytesperline, bytesperline))
    else:
        data = data.reshape((h, len(data) // h))

    # cut the extra padding on the right
    extra = bytesperline - w * bitspp // 8
    if extra:
        data = np.delete(data, np.s_[-extra:], 1)

    arr16 = data.astype(np.uint16) << np.uint16(2)
    for byte in range(4):
        asd = (arr16[:, 4::5] >> ((4 - byte) * 2)) & 0b11
        arr16[:, byte::5] |= asd
    arr16 = np.delete(arr16, np.s_[4::5], 1)

    idx = bayer_pattern.find('R')
    assert idx != -1
    r0 = (idx % 2, idx // 2)

    idx = bayer_pattern.find('G')
    assert idx != -1
    g0 = (idx % 2, idx // 2)

    idx = bayer_pattern.find('G', idx + 1)
    assert idx != -1
    g1 = (idx % 2, idx // 2)

    idx = bayer_pattern.find('B')
    assert idx != -1
    b0 = (idx % 2, idx // 2)

    rgb = demosaic(arr16, r0, g0, g1, b0)
    rgb = (rgb >> (bitspp - 8)).astype(np.uint8)  # pyright: ignore [reportOperatorIssue]

    return rgb


def raw_to_bgr888(data: npt.NDArray[np.uint8], w, h, bytesperline, fmt: PixelFormat):
    fmtname = fmt.name

    bayer_pattern = fmtname[1:5]

    packed = fmtname.endswith('P')

    if packed:
        bitspp = int(fmtname[5:-1])
    else:
        bitspp = int(fmtname[5:])

    if packed:
        return raw_packed_to_bgr888(data, w, h, bytesperline, fmt)

    if bitspp == 8:
        arr16 = data.reshape((h, w))
        arr16 = arr16.astype(np.uint16)
    elif bitspp in [10, 12, 16]:
        arr16 = data.view(np.uint16)
        arr16 = arr16.reshape((h, w))
    else:
        raise RuntimeError('Bad bitspp:' + str(bitspp))

    idx = bayer_pattern.find('R')
    assert idx != -1
    r0 = (idx % 2, idx // 2)

    idx = bayer_pattern.find('G')
    assert idx != -1
    g0 = (idx % 2, idx // 2)

    idx = bayer_pattern.find('G', idx + 1)
    assert idx != -1
    g1 = (idx % 2, idx // 2)

    idx = bayer_pattern.find('B')
    assert idx != -1
    b0 = (idx % 2, idx // 2)

    rgb = demosaic(arr16, r0, g0, g1, b0)
    rgb = (rgb >> (bitspp - 8)).astype(np.uint8)  # pyright: ignore [reportOperatorIssue]

    return rgb
