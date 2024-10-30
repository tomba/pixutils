#!/usr/bin/python3

import os
import gzip
import unittest
import numpy as np

from pixutils.formats import PixelFormats, str_to_fourcc
from pixutils.conv import buffer_to_bgr888

TEST_PATH = os.path.dirname(os.path.abspath(__file__))


class TestConv(unittest.TestCase):
    def test_conversions(self):
        # Some formats are not supported yet

        fmts = [
            #'BG16',
            'BG24',
            #'BX24',
            'NV12',
            #'NV21',
            #'RG16',
            'RG24',
            #'RX24',
            'UYVY',
            'XB24',
            'XR24',
            'YUYV',
        ]

        # fmts = [ 'YUYV' ]

        w = 640
        h = 480

        fname = f'{TEST_PATH}/data/test-{w}-{h}-BG24.bin.gz'

        with gzip.open(fname, 'rb') as f:
            ref_buf = np.frombuffer(f.read(), dtype=np.uint8)
        ref = buffer_to_bgr888(PixelFormats.BGR888, w, h, 0, ref_buf)
        ref = ref.astype(np.int16)

        for fourccstr in fmts:
            fmt = PixelFormats.find_drm_fourcc(str_to_fourcc(fourccstr))

            bytesperline = 0  # fmt.stride(w)

            fname = f'{TEST_PATH}/data/test-{w}-{h}-{fourccstr}.bin.gz'

            with gzip.open(fname, 'rb') as f:
                data_in = np.frombuffer(f.read(), dtype=np.uint8)

            # Note: the yuv test images are in bt601 limited
            options = {
                'range': 'limited',
                'encoding': 'bt601',
            }

            rgb = buffer_to_bgr888(fmt, w, h, bytesperline, data_in, options)

            self.assertEqual(rgb.shape, ref.shape)

            rgb = rgb.astype(np.int16)

            diff = rgb - ref

            # Exact match?
            if not diff.any():
                continue

            diff = abs(diff)

            b = diff[:, :, 0]
            g = diff[:, :, 1]
            r = diff[:, :, 2]

            # 2.5 is an arbitrary number that seems to pass for now
            self.assertLessEqual(b.mean(), 2.5)
            self.assertLessEqual(g.mean(), 2.5)
            self.assertLessEqual(r.mean(), 2.5)


if __name__ == '__main__':
    unittest.main()
