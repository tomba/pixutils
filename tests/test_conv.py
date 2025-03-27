#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import unittest
import numpy as np

from pixutils.formats import PixelFormats, PixelFormat
from pixutils.conv import buffer_to_bgr888

WIDTH = 640
HEIGHT = 480

SEED = 1234

FMTS = {
    PixelFormats.BGR888: (
        'dc1f1d9cf96911bc33682edcf9b81af2b3184d88890aa118c09a7ba0932826c2',
        'dc1f1d9cf96911bc33682edcf9b81af2b3184d88890aa118c09a7ba0932826c2',
    ),
    PixelFormats.RGB888: (
        'dc1f1d9cf96911bc33682edcf9b81af2b3184d88890aa118c09a7ba0932826c2',
        '9c2ced2d2d19197a6f49b513dcc5a0aa02837bb1876c5f8629820e998cafadad',
    ),
    PixelFormats.XBGR8888: (
        'd555e8545b743011df3c27f0a7056552b302a9089b7b6d8d9562402c38c40f1e',
        '366abaf8289cef57ee44410a7468189ece1040821073df18255e126196fad03d',
    ),
    PixelFormats.XRGB8888: (
        'd555e8545b743011df3c27f0a7056552b302a9089b7b6d8d9562402c38c40f1e',
        '5cd101d48be291acd03a7ce2443cebb081f9d6489c8213960b8fbb49274e3e3b',
    ),
    PixelFormats.NV12: (
        '7d75356fc0ab885264f4ba453fc0c9eec23b70412fd3a8c261eb0d7ed5a1ea77',
        'fbca6e74ae93a50aac8233b80a19fc5dd3ee0347c61ad27012f7ad3e34f03d62',
    ),
    PixelFormats.UYVY: (
        '50e4efdafe1d8c6cc2b31fb2d9c1be2fa77363d6bd759417cab15ec580ce0f19',
        '5c8aaa1acf732e1a126fdc815b595197d02830ebc743077d37a92bef8c40f8fa',
    ),
    PixelFormats.YUYV: (
        '50e4efdafe1d8c6cc2b31fb2d9c1be2fa77363d6bd759417cab15ec580ce0f19',
        '73cdd1de32cbc4ff9d1d8d03649b83ab4ccf5ef865cb3d0c1183de3752a0b727',
    ),
    PixelFormats.SRGGB10: (
        '3eafcda7e182be78032d6296002c2a4780ce8317a67f30578453b97d24dd3205',
        '59b206fa98863196fb3b8276b43aa9e54d4b6826b6e64d3c21bc76ce8914cf89',
    ),
    PixelFormats.SRGGB10P: (
        'ab7ddc194c770396ae961630df80e05cfaf63a7de93fb19ff49aef999234310d',
        '9603db169d70d12f8495fad8efac4a15d141ab8964803cd4f5ad7cce430ca42b',
    ),
    PixelFormats.SRGGB12: (
        'dccb1922642869f266ed341b27200dca58add54c344c69a8991cedf6232fbb33',
        '49999659bafe57a0e90672f176ccb06f6d14dfe67c84a1f1a44249d2513c24a9',
    ),
    PixelFormats.SRGGB16: (
        '50e4efdafe1d8c6cc2b31fb2d9c1be2fa77363d6bd759417cab15ec580ce0f19',
        '93b4137b70cccb9f45ff78cabef0406c2150e5ad7ac708dbd4aaaddf42dff15b',
    ),
    PixelFormats.SRGGB8: (
        '0617515ed5db0a0ce1945ddd1887d7616137055d424199eddc71dceece53a740',
        '4ebaac89404d809284c4ca4beab07dc6f77a8d8250cbc7cf225d13f40b8579be',
    ),
}


def generate_test_buffer(fmt: PixelFormat):
    rnd = np.random.Generator(np.random.PCG64(SEED))

    size = fmt.framesize(WIDTH, HEIGHT)

    buf = np.frombuffer(rnd.bytes(size), dtype=np.uint8)

    # Mask out the padding bits

    if fmt == PixelFormats.SRGGB10:
        buf = buf.view(np.uint16)
        buf = buf & ((1 << 10) - 1)
    elif fmt == PixelFormats.SRGGB12:
        buf = buf.view(np.uint16)
        buf = buf & ((1 << 12) - 1)
    elif fmt == PixelFormats.XBGR8888:
        buf = buf.view(np.uint32)
        buf = buf & ((1 << 24) - 1)
    elif fmt == PixelFormats.XRGB8888:
        buf = buf.view(np.uint32)
        buf = buf & ((1 << 24) - 1)

    return buf


def generate_test_data_dict():
    print('FMTS = {')
    for fmt in FMTS.keys():
        src_buf = generate_test_buffer(fmt)

        options = {
            'range': 'limited',
            'encoding': 'bt601',
        }

        rgb_buf = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, src_buf, options)

        src_sha = hashlib.sha256(src_buf.tobytes()).hexdigest()
        rgb_sha = hashlib.sha256(rgb_buf.tobytes()).hexdigest()

        print(f"    PixelFormats.{fmt.name}: ('{src_sha}', '{rgb_sha}'),")
    print('}')


def save_test_data():
    for fmt in FMTS.keys():
        src_buf = generate_test_buffer(fmt)

        options = {
            'range': 'limited',
            'encoding': 'bt601',
        }

        rgb_buf = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, src_buf, options)

        src_file = f'{WIDTH}x{HEIGHT}-{fmt}.bin.gz'
        rgb_file = f'{WIDTH}x{HEIGHT}-{fmt}-BGR888.bin.gz'

        with open(src_file, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
                gz.write(src_buf.tobytes())

        with open(rgb_file, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
                gz.write(rgb_buf.tobytes())


class TestConv(unittest.TestCase):
    # Test functions added dynamically below
    pass


def create_test_function(fmt: PixelFormat, ref_src_sha: str, ref_rgb_sha: str):
    def test_function(self):
        src_buf = generate_test_buffer(fmt)

        options = {
            'range': 'limited',
            'encoding': 'bt601',
        }

        rgb_buf = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, src_buf, options)

        src_sha = hashlib.sha256(src_buf.tobytes()).hexdigest()
        rgb_sha = hashlib.sha256(rgb_buf.tobytes()).hexdigest()

        self.assertEqual(src_sha, ref_src_sha, f'SHA mismatch for {fmt.name} source')
        self.assertEqual(rgb_sha, ref_rgb_sha, f'SHA mismatch for {fmt.name} RGB')

    return test_function

for fmt, (ref_src_sha, ref_rgb_sha) in FMTS.items():
    test_name = f'test_conv_{fmt.name}'
    test = create_test_function(fmt, ref_src_sha, ref_rgb_sha)
    setattr(TestConv, test_name, test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save', action='store_true', help='Generate frames, save to files and exit.')
    parser.add_argument('--dict', action='store_true', help='Generate FMTS dict, print and exit.')
    args, unknown = parser.parse_known_args()

    if args.save:
        save_test_data()
    elif args.dict:
        generate_test_data_dict()
    else:
        unittest.main()
