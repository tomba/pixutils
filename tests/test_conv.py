#!/usr/bin/env python3

import glob
import gzip
import os
import unittest
import re
import numpy as np

from pixutils.formats import PixelFormats, PixelFormat
from pixutils.conv import buffer_to_bgr888

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = f'{TEST_PATH}/conv-test-data'


class TestConv(unittest.TestCase):
    def test_conversions(self):
        for fname in glob.glob(f'{DATA_PATH}/*.bin.gz'):
            bname = os.path.basename(fname)
            m = re.match(r'(\d+)x(\d+)-(\w+).bin.gz',  bname)
            if not m:
                continue

            width = int(m.group(1))
            height = int(m.group(2))
            fmt = PixelFormats.find_by_name(m.group(3))

            self.run_test_image(width, height, fmt)

    def run_test_image(self, width: int, height: int, fmt: PixelFormat):
        with gzip.open(f'{DATA_PATH}/{width}x{height}-{fmt.name}.bin.gz', 'rb') as f:
            src_buf = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(f'{DATA_PATH}/{width}x{height}-{fmt.name}-BGR888.bin.gz', 'rb') as f:
            ref_buf = np.frombuffer(f.read(), dtype=np.uint8)

        options = {
            'range': 'limited',
            'encoding': 'bt601',
        }

        rgb_buf = buffer_to_bgr888(fmt, width, height, 0, src_buf, options)
        rgb_buf = rgb_buf.flatten()

        self.assertEqual(rgb_buf.shape, ref_buf.shape)

        diff = rgb_buf - ref_buf

        self.assertFalse(diff.any())


if __name__ == '__main__':
    unittest.main()
