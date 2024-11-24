# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2023, Tomi Valkeinen <tomi.valkeinen@ideasonboard.com>

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pixutils.formats import PixelFormat

__all__ = ['raw_to_bgr888']


@dataclass
class BayerPattern:
    """Represents a Bayer pattern configuration"""

    r0: tuple[int, int]
    g0: tuple[int, int]
    g1: tuple[int, int]
    b0: tuple[int, int]

    @classmethod
    def from_pattern(cls, pattern: str):
        """Parse a Bayer pattern string (e.g., 'SRGGB') into coordinates"""
        idx = pattern.find('R')
        r0 = (idx % 2, idx // 2)

        idx = pattern.find('G')
        g0 = (idx % 2, idx // 2)

        idx = pattern.find('G', idx + 1)
        g1 = (idx % 2, idx // 2)

        idx = pattern.find('B')
        b0 = (idx % 2, idx // 2)

        return cls(r0, g0, g1, b0)


@dataclass
class RawFormat:
    """Represents a raw image format configuration"""

    bayer_pattern: BayerPattern
    bits_per_pixel: int
    is_packed: bool

    @classmethod
    def from_pixelformat(cls, fmt: PixelFormat):
        """Parse a PixelFormat into raw format configuration"""
        name = fmt.name
        pattern = name[1:5]  # e.g., 'RGGB' from 'SRGGB8'
        is_packed = name.endswith('P')

        if is_packed:
            bpp = int(name[5:-1])  # Remove 'P' for packed formats
        else:
            bpp = int(name[5:])  # Direct BPP value

        return cls(bayer_pattern=BayerPattern.from_pattern(pattern),
                   bits_per_pixel=bpp,
                   is_packed=is_packed)


def prepare_packed_raw(data: npt.NDArray[np.uint8], width: int, height: int,
                       bits_per_pixel: int, bytesperline: int) -> npt.NDArray[np.uint16]:
    # Only 10 bpp is supported for now
    assert bits_per_pixel == 10

    # Reshape into rows if bytesperline is provided
    if bytesperline:
        data = data.reshape((len(data) // bytesperline, bytesperline))
    else:
        data = data.reshape((height, len(data) // height))

    # Remove padding if present
    padded_width = width * 10 // 8  # For 10-bit packed
    if bytesperline > padded_width:
        data = np.delete(data, np.s_[padded_width:], 1)

    # Convert to 16-bit and handle packing
    arr16 = data.astype(np.uint16) << np.uint16(2)
    for byte in range(4):
        asd = (arr16[:, 4::5] >> ((4 - byte) * 2)) & 0b11
        arr16[:, byte::5] |= asd
    arr16 = np.delete(arr16, np.s_[4::5], 1)

    return arr16


def prepare_unpacked_raw(data: npt.NDArray[np.uint8], width: int, height: int,
                        bits_per_pixel: int) -> npt.NDArray[np.uint16]:
    if bits_per_pixel == 8:
        return data.reshape((height, width)).astype(np.uint16)

    if bits_per_pixel in [10, 12, 16]:
        return data.view(np.uint16).reshape((height, width))

    raise RuntimeError(f'Unsupported bits per pixel: {bits_per_pixel}')


def demosaic(data: npt.NDArray[np.uint16], pattern: BayerPattern) -> npt.NDArray[np.uint16]:
    # Debayering code from PiCamera documentation

    # Separate the components from the Bayer data to RGB planes
    h, w = data.shape

    # Separate the components from the Bayer data to RGB planes
    rgb = np.zeros((h, w, 3), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[pattern.r0[1] :: 2, pattern.r0[0] :: 2]  # Red
    rgb[0::2, 0::2, 1] = data[pattern.g0[1] :: 2, pattern.g0[0] :: 2]  # Green
    rgb[1::2, 1::2, 1] = data[pattern.g1[1] :: 2, pattern.g1[0] :: 2]  # Green
    rgb[0::2, 1::2, 2] = data[pattern.b0[1] :: 2, pattern.b0[0] :: 2]  # Blue

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

    rgb = np.pad(rgb, [
        (border[0], border[0]),
        (border[1], border[1]),
        (0, 0),
    ], 'constant')
    bayer = np.pad(bayer, [
        (border[0], border[0]),
        (border[1], border[1]),
        (0, 0),
    ], 'constant')

    # For each plane in the RGB data, we calculate the 3x3 window sum
    # and divide it with the weighted average. This version uses direct
    # computation of the sum, instead of using numpy's as_strided()
    # and einsum(), as the direct version is 2x as fast.

    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]

        # Direct computation of 3x3 window sum
        psum = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])

        bsum = (b[:-2, :-2] + b[:-2, 1:-1] + b[:-2, 2:] +
                b[1:-1, :-2] + b[1:-1, 1:-1] + b[1:-1, 2:] +
                b[2:, :-2] + b[2:, 1:-1] + b[2:, 2:])

        output[..., plane] = psum // bsum

    return output


def raw_to_bgr888(data: npt.NDArray[np.uint8], width: int, height: int,
                  bytesperline: int, fmt: PixelFormat) -> npt.NDArray[np.uint8]:
    # Parse the format
    raw_fmt = RawFormat.from_pixelformat(fmt)

    # Prepare the raw data into a common 16-bit format
    if raw_fmt.is_packed:
        arr16 = prepare_packed_raw(data, width, height, raw_fmt.bits_per_pixel,
                                   bytesperline)
    else:
        arr16 = prepare_unpacked_raw(data, width, height, raw_fmt.bits_per_pixel)

    # Perform demosaic
    rgb = demosaic(arr16, raw_fmt.bayer_pattern)

    # Convert to 8-bit BGR
    return (rgb >> (raw_fmt.bits_per_pixel - 8)).astype(np.uint8)
