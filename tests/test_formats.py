#!/usr/bin/env python3

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import NamedTuple

import pytest

from pixutils.formats.pixelformats import PixelFormat, PixelFormats


class FormatTestData(NamedTuple):
    format: PixelFormat
    width: int
    height: int
    strides: list[int]
    sizes: list[int]


TEST_DATA = [
    FormatTestData(
        format=PixelFormats.XRGB8888,
        width=1920,
        height=1080,
        strides=[1920 * 4],
        sizes=[1920 * 4 * 1080],
    ),
    FormatTestData(
        format=PixelFormats.YUYV,
        width=1920,
        height=1080,
        strides=[1920 * 2],
        sizes=[1920 * 2 * 1080],
    ),
    FormatTestData(
        format=PixelFormats.NV12,
        width=1920,
        height=1080,
        strides=[1920 * 1, 1920 * 2 // 2],
        sizes=[1920 * 1 * 1080, 1920 * 2 // 2 * 1080 // 2],
    ),
    FormatTestData(
        format=PixelFormats.NV16,
        width=1920,
        height=1080,
        strides=[1920 * 1, 1920 * 2 // 2],
        sizes=[1920 * 1 * 1080, 1920 * 2 // 2 * 1080 // 1],
    ),
    FormatTestData(
        format=PixelFormats.P030,
        width=1920,
        height=1080,
        strides=[1920 // 3 * 4, 1920 // 3 * 8 // 2],
        sizes=[1920 // 3 * 4 * 1080, 1920 // 3 * 8 // 2 * 1080 // 2],
    ),
    FormatTestData(
        format=PixelFormats.P230,
        width=1920,
        height=1080,
        strides=[1920 // 3 * 4, 1920 // 3 * 8 // 2],
        sizes=[1920 // 3 * 4 * 1080, 1920 // 3 * 8 // 2 * 1080 // 1],
    ),
    FormatTestData(
        format=PixelFormats.SBGGR8,
        width=1920,
        height=1080,
        strides=[1920 * 1],
        sizes=[1920 * 1 * 1080],
    ),
    FormatTestData(
        format=PixelFormats.SRGGB10,
        width=1920,
        height=1080,
        strides=[1920 * 2],
        sizes=[1920 * 2 * 1080],
    ),
    FormatTestData(
        format=PixelFormats.SRGGB10P,
        width=1920,
        height=1080,
        strides=[1920 * 5 // 4],
        sizes=[1920 * 5 // 4 * 1080],
    ),
]


@pytest.mark.parametrize('data', TEST_DATA, ids=lambda d: d.format.name)
def test_plane_strides_and_sizes(data: FormatTestData):
    fmt = data.format
    for idx, _ in enumerate(fmt.planes):
        stride = fmt.stride(data.width, idx)
        size = fmt.planesize(stride, data.height, idx)
        assert stride == data.strides[idx], f'stride failed for {fmt}'
        assert size == data.sizes[idx], f'size failed for {fmt}'

        dumb_size = reduce(mul, fmt.dumb_size(data.width, data.height, idx)) // 8
        assert size == dumb_size, f'dumb size failed for {fmt}'


_MULTIPLANE_FORMATS = [f for f in PixelFormats.get_formats() if len(f.planes) >= 2]


@pytest.mark.parametrize('fmt', _MULTIPLANE_FORMATS)
def test_extrapolate_matches_natural_stride(fmt: PixelFormat):
    # For every planar/semi-planar format, extrapolating from plane-0's
    # natural stride must reproduce the natural stride of every other plane.
    width = 1920
    s0 = fmt.stride(width, 0)
    for i in range(len(fmt.planes)):
        assert fmt.extrapolate_stride(s0, i) == fmt.stride(width, i), (
            f'extrapolation mismatch for {fmt.name} plane {i}'
        )


def test_extrapolate_preserves_padding_ratio():
    # NV12: chroma stride equals luma stride (ratio 1), padding carries over.
    nv12 = PixelFormats.NV12
    assert nv12.extrapolate_stride(1920, 1) == 1920
    assert nv12.extrapolate_stride(2048, 1) == 2048

    # YUV420: chroma stride is half of luma stride; even padding halves.
    yuv420 = PixelFormats.YUV420
    assert yuv420.extrapolate_stride(1920, 1) == 960
    assert yuv420.extrapolate_stride(2048, 1) == 1024
    assert yuv420.extrapolate_stride(2048, 2) == 1024


def test_extrapolate_plane_zero_returns_input():
    assert PixelFormats.NV12.extrapolate_stride(1921, 0) == 1921


def test_extrapolate_raises_on_non_integer_result():
    # YUV420 chroma has hsub=2, so an odd plane-0 stride cannot be halved.
    with pytest.raises(ValueError):
        PixelFormats.YUV420.extrapolate_stride(1921, 1)


def test_extrapolate_raises_on_invalid_plane_index():
    with pytest.raises(RuntimeError):
        PixelFormats.NV12.extrapolate_stride(1920, 2)
