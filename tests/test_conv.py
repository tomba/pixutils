#!/usr/bin/python3

import os
import unittest
import numpy as np

from pixutils import PixelFormats
from pixutils.conv.conv import to_bgr888

TEST_PATH = os.path.dirname(os.path.abspath(__file__))

class TestRGBConv(unittest.TestCase):
    def test_rgb(self):
        tests = [
            (PixelFormats.BGR888, 2, 2, 0,
             (1, 2, 3, 4, 5, 6,
              1, 2, 3, 4, 5, 6),
             (1, 2, 3, 4, 5, 6,
              1, 2, 3, 4, 5, 6))
            #(PixelFormats.RGB888, 64, 48, 0, "rgb888-in.raw", "rgb888-out.raw"),
        ]

        for fmt, w, h, bytesperline, data_in, data_ref in tests:
            if isinstance(data_in, str):
                data_in = np.fromfile(TEST_PATH + '/' + data_in, dtype=np.uint8)
            elif isinstance(data_in, (list, tuple)):
                data_in = np.array(data_in, dtype=np.uint8)

            if isinstance(data_ref, str):
                data_ref = np.fromfile(TEST_PATH + '/' + data_ref, dtype=np.uint8)
            elif isinstance(data_ref, (list, tuple)):
                data_ref = np.array(data_ref, dtype=np.uint8)

            data_out = to_bgr888(fmt, w, h, bytesperline, data_in)

            # Flatten for comparison
            data_out = data_out.flatten()

            self.assertTrue(np.array_equal(data_out, data_ref))

if __name__ == '__main__':
    unittest.main()
