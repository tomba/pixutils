#!/usr/bin/env python3

from __future__ import annotations

import unittest
from functools import reduce
from operator import mul
from typing import NamedTuple

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


class TestFormats(unittest.TestCase):
    def test_formats(self):
        for data in TEST_DATA:
            self.run_data(data)

    def run_data(self, data: FormatTestData):
        fmt = data.format
        for idx, _ in enumerate(fmt.planes):
            stride = fmt.stride(data.width, idx)
            size = fmt.planesize(stride, data.height, idx)
            self.assertEqual(stride, data.strides[idx], f'stride failed for {fmt}')
            self.assertEqual(size, data.sizes[idx], f'size failed for {fmt}')

            dumb_size = reduce(mul, fmt.dumb_size(data.width, data.height, idx)) // 8

            self.assertEqual(size, dumb_size, f'dumb size failed for {fmt}')


class TestExtrapolateStride(unittest.TestCase):
    def test_matches_natural_stride(self):
        # For every planar/semi-planar format, extrapolating from plane-0's
        # natural stride must reproduce the natural stride of every other plane.
        width = 1920
        for fmt in PixelFormats.get_formats():
            if len(fmt.planes) < 2:
                continue
            s0 = fmt.stride(width, 0)
            for i in range(len(fmt.planes)):
                self.assertEqual(
                    fmt.extrapolate_stride(s0, i),
                    fmt.stride(width, i),
                    f'extrapolation mismatch for {fmt.name} plane {i}',
                )

    def test_preserves_padding_ratio(self):
        # NV12: chroma stride equals luma stride (ratio 1), padding carries over.
        nv12 = PixelFormats.NV12
        self.assertEqual(nv12.extrapolate_stride(1920, 1), 1920)
        self.assertEqual(nv12.extrapolate_stride(2048, 1), 2048)

        # YUV420: chroma stride is half of luma stride; even padding halves.
        yuv420 = PixelFormats.YUV420
        self.assertEqual(yuv420.extrapolate_stride(1920, 1), 960)
        self.assertEqual(yuv420.extrapolate_stride(2048, 1), 1024)
        self.assertEqual(yuv420.extrapolate_stride(2048, 2), 1024)

    def test_plane_zero_returns_input(self):
        self.assertEqual(PixelFormats.NV12.extrapolate_stride(1921, 0), 1921)

    def test_raises_on_non_integer_result(self):
        # YUV420 chroma has hsub=2, so an odd plane-0 stride cannot be halved.
        with self.assertRaises(ValueError):
            PixelFormats.YUV420.extrapolate_stride(1921, 1)

    def test_raises_on_invalid_plane_index(self):
        with self.assertRaises(RuntimeError):
            PixelFormats.NV12.extrapolate_stride(1920, 2)


if __name__ == '__main__':
    unittest.main()
