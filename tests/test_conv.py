#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import unittest
import numpy as np

from pixutils.formats import PixelFormat, PixelColorEncoding
from pixutils.conv import buffer_to_bgr888
from test_conv_data import FMTS

WIDTH = 640
HEIGHT = 480

SEED = 1234


def get_bit_mask(fmt: PixelFormat):
    """Returns (dtype, mask) tuple for masking padding bits, or None if no masking needed."""
    # RAW 10-bit formats (SRGGB10, SBGGR10, SGRBG10, SGBRG10)
    if fmt.name.endswith('10') and fmt.color == PixelColorEncoding.RAW and not fmt.packed:
        return (np.uint16, (1 << 10) - 1)
    # RAW 12-bit formats (SRGGB12, SBGGR12, SGRBG12, SGBRG12)
    elif fmt.name.endswith('12') and fmt.color == PixelColorEncoding.RAW and not fmt.packed:
        return (np.uint16, (1 << 12) - 1)
    # XBGR8888, XRGB8888 formats - mask out alpha channel
    elif fmt.name in ('XBGR8888', 'XRGB8888'):
        return (np.uint32, (1 << 24) - 1)
    return None



def generate_test_buffer(fmt: PixelFormat):
    rnd = np.random.Generator(np.random.PCG64(SEED))

    size = fmt.framesize(WIDTH, HEIGHT)

    buf = np.frombuffer(rnd.bytes(size), dtype=np.uint8)

    # Mask out the padding bits
    mask_info = get_bit_mask(fmt)
    if mask_info:
        dtype, mask = mask_info
        buf = buf.view(dtype)
        buf = buf & mask

    return buf


def generate_test_data_dict():
    print('#!/usr/bin/env python3')
    print()
    print('from pixutils.formats import PixelFormats')
    print()
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

        print(f'    PixelFormats.{fmt.name}: (')
        print(f"        '{src_sha}',")
        print(f"        '{rgb_sha}',")
        print('    ),')
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

# Create test methods dynamically at module level for unittest discovery
for fmt, (ref_src_sha, ref_rgb_sha) in FMTS.items():
    test_name = f'test_conv_{fmt.name}'
    test = create_test_function(fmt, ref_src_sha, ref_rgb_sha)
    setattr(TestConv, test_name, test)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save', action='store_true', help='Generate frames, save to files and exit.')
    parser.add_argument('--dict', action='store_true', help='Generate FMTS dict, print and exit.')
    args, _ = parser.parse_known_args()

    if args.save:
        save_test_data()
    elif args.dict:
        generate_test_data_dict()
    else:
        unittest.main()

if __name__ == '__main__':
    main()
