#!/usr/bin/env python3

import argparse
import time

import numpy as np

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888


def main():
    parser = argparse.ArgumentParser(description='Test conversion performance.')
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')
    parser.add_argument('-f', '--format', type=str, default='XRGB8888', help='Pixel format')
    parser.add_argument('-l', '--loops', type=int, default=100, help='Number of loops')
    parser.add_argument('--stride', type=int, default=0, help='Stride')
    args = parser.parse_args()

    fmt = PixelFormats.find_by_name(args.format)

    # Only single plane formats are supported
    assert len(fmt.planes) == 1

    stride = args.stride if args.stride > 0 else fmt.stride(args.width)
    planesize = fmt.planesize(stride, args.height)
    size = planesize
    buf = np.zeros(size, dtype=np.uint8)

    print(f'Image size: {args.width}x{args.height}, format: {args.format}, stride: {stride}, size {size}')

    options = {
        'range': 'limited',
        'encoding': 'bt601',
    }

    # Warmup run
    buffer_to_bgr888(fmt, args.width, args.height, stride, buf, options)

    t1 = time.monotonic()

    for _ in range(args.loops):
        buffer_to_bgr888(fmt, args.width, args.height, stride, buf, options)

    t2 = time.monotonic()
    print(f'{args.loops} loops took {(t2 - t1) * 1000:.3f} ms')

if __name__ == '__main__':
    main()
