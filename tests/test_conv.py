#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import unittest
import numpy as np

from pixutils.formats import PixelFormat, PixelColorEncoding, PixelFormats
from pixutils.conv import buffer_to_bgr888
from test_conv_data import FMTS  # type: ignore[import-not-found]

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


def generate_test_data():
    import sys

    print('from pixutils.formats import PixelFormats')
    print('from conv_test_case import ConvTestCase  # type: ignore[import-not-found]')
    print()
    print('FMTS = [')

    def is_yuv_format(pixel_format):
        return pixel_format.color == PixelColorEncoding.YUV

    def is_luma_only_format(pixel_format):
        return pixel_format == PixelFormats.Y8

    def generate_test_case(pixel_format, options=None):
        try:
            src_buf = generate_test_buffer(pixel_format)
            rgb_buf = buffer_to_bgr888(pixel_format, WIDTH, HEIGHT, 0, src_buf, options or {})

            src_sha = hashlib.sha256(src_buf.tobytes()).hexdigest()
            rgb_sha = hashlib.sha256(rgb_buf.tobytes()).hexdigest()

            return (src_sha, rgb_sha)
        except Exception as e:
            print(f'# Skipping {pixel_format.name} with options {options}: {e}', file=sys.stderr)
            return None

    # Group formats by type
    rgb_bayer_formats = []
    yuv_formats = []

    for pixel_format in PixelFormats.get_formats():
        if is_yuv_format(pixel_format):
            yuv_formats.append(pixel_format)
        else:
            rgb_bayer_formats.append(pixel_format)

    # Process RGB/Bayer formats (no options)
    print('    # RGB/Bayer formats - conversion not affected by YUV options')
    for pixel_format in rgb_bayer_formats:
        result = generate_test_case(pixel_format)
        if result:
            src_sha, rgb_sha = result
            print(f'    ConvTestCase(PixelFormats.{pixel_format.name},')
            print(f"        '{src_sha}',")
            print(f"        '{rgb_sha}'),")

    # Process YUV formats (with range/encoding combinations)
    if yuv_formats:
        print()
        print('    # YUV formats - conversion affected by range/encoding options')

        ranges = ['limited', 'full']
        encodings = ['bt601', 'bt709', 'bt2020']

        for pixel_format in yuv_formats:
            if is_luma_only_format(pixel_format):
                # Luma-only formats: only test range variations, encoding doesn't apply
                for range_val in ranges:
                    options = {'range': range_val}
                    result = generate_test_case(pixel_format, options)
                    if result:
                        src_sha, rgb_sha = result
                        print(f'    ConvTestCase(PixelFormats.{pixel_format.name},')
                        print(f"        '{src_sha}',")
                        print(f"        '{rgb_sha}',")
                        print(f'        {options}),')
            else:
                # Full YUV formats: test all range×encoding combinations
                for range_val in ranges:
                    for encoding in encodings:
                        options = {'range': range_val, 'encoding': encoding}
                        result = generate_test_case(pixel_format, options)
                        if result:
                            src_sha, rgb_sha = result
                            print(f'    ConvTestCase(PixelFormats.{pixel_format.name},')
                            print(f"        '{src_sha}',")
                            print(f"        '{rgb_sha}',")
                            print(f'        {options}),')

    print(']')


def save_test_data():
    for test_case in FMTS:
        src_buf = generate_test_buffer(test_case.pixel_format)
        rgb_buf = buffer_to_bgr888(test_case.pixel_format, WIDTH, HEIGHT, 0, src_buf, test_case.options)

        # Use test_case description for file naming to handle options
        base_name = f'{WIDTH}x{HEIGHT}-{test_case.description}'
        src_file = f'{base_name}.bin.gz'
        rgb_file = f'{base_name}-BGR888.bin.gz'

        with open(src_file, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
                gz.write(src_buf.tobytes())

        with open(rgb_file, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
                gz.write(rgb_buf.tobytes())


class TestConv(unittest.TestCase):
    # Test functions added dynamically below
    pass


def create_test_function(test_case):
    def test_function(self):
        src_buf = generate_test_buffer(test_case.pixel_format)
        rgb_buf = buffer_to_bgr888(test_case.pixel_format, WIDTH, HEIGHT, 0, src_buf, test_case.options)

        src_sha = hashlib.sha256(src_buf.tobytes()).hexdigest()
        rgb_sha = hashlib.sha256(rgb_buf.tobytes()).hexdigest()

        self.assertEqual(src_sha, test_case.src_sha, f'SHA mismatch for {test_case.description} source')
        self.assertEqual(rgb_sha, test_case.rgb_sha, f'SHA mismatch for {test_case.description} RGB')

    return test_function

# Create test methods dynamically at module level for unittest discovery
for test_case in FMTS:
    test_name = f'test_conv_{test_case.description}'
    test = create_test_function(test_case)
    setattr(TestConv, test_name, test)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save', action='store_true', help='Generate frames, save to files and exit.')
    parser.add_argument('--generate-data', action='store_true', help='Generate FMTS list, print and exit.')
    args, _ = parser.parse_known_args()

    if args.save:
        save_test_data()
    elif args.generate_data:
        generate_test_data()
    else:
        unittest.main()

if __name__ == '__main__':
    main()
