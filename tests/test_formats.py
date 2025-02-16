#!/usr/bin/python3

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import NamedTuple
import unittest

from pixutils.formats.pixelformats import PixelFormat, PixelFormats

class TestData(NamedTuple):
    format: PixelFormat
    width: int
    height: int
    strides: list[int]
    sizes: list[int]


TEST_DATA = [
    TestData(format=PixelFormats.XRGB8888,
             width=1920, height=1080,
             strides=[1920*4], sizes=[1920*4*1080]),

    TestData(format=PixelFormats.YUYV,
             width=1920, height=1080,
             strides=[1920*2], sizes=[1920*2*1080]),

    TestData(format=PixelFormats.NV12,
             width=1920, height=1080,
             strides=[1920*1, 1920*2//2], sizes=[1920*1*1080, 1920*2//2 * 1080//2]),

    TestData(format=PixelFormats.NV16,
             width=1920, height=1080,
             strides=[1920*1, 1920*2//2], sizes=[1920*1*1080, 1920*2//2 * 1080//1]),

    TestData(format=PixelFormats.SBGGR8,
             width=1920, height=1080,
             strides=[1920*1], sizes=[1920*1*1080]),

    TestData(format=PixelFormats.SRGGB10,
             width=1920, height=1080,
             strides=[1920*2], sizes=[1920*2*1080]),

    TestData(format=PixelFormats.SRGGB10P,
             width=1920, height=1080,
             strides=[1920*5//4], sizes=[1920*5//4*1080]),
]

class TestFormats(unittest.TestCase):
    def test_formats(self):
        for data in TEST_DATA:
            self.run_data(data)

    def run_data(self, data: TestData):
        fmt = data.format
        for idx, _ in enumerate(fmt.planes):
            stride = fmt.stride(data.width, idx)
            size = fmt.planesize(stride, data.height, idx)
            self.assertEqual(stride,
                             data.strides[idx],
                             f'stride failed for {fmt}')
            self.assertEqual(size,
                             data.sizes[idx],
                             f'size failed for {fmt}')

            dumb_size = reduce(mul, fmt.dumb_size(data.width, data.height, idx)) // 8

            self.assertEqual(size,
                             dumb_size,
                             f'dumb size failed for {fmt}')


if __name__ == '__main__':
    unittest.main()
