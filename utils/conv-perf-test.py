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

    # Drop this when stride works
    if len(fmt.planes) > 1 and args.stride > 0:
        raise ValueError('Custom stride is not supported with multiplanar formats')

    # Calculate total buffer size for all planes
    if args.stride > 0:
        # Single plane format with custom stride
        size = fmt.planesize(args.stride, args.height, 0)
    else:
        # Use framesize for both single and multiplanar formats
        size = fmt.framesize(args.width, args.height)

    buf = np.zeros(size, dtype=np.uint8)

    stride = args.stride if args.stride > 0 else fmt.stride(args.width, 0)

    options = {
        'range': 'limited',
        'encoding': 'bt601',
    }

    bytesperline = 0 if len(fmt.planes) > 1 else stride

    # Warmup run
    buffer_to_bgr888(fmt, args.width, args.height, bytesperline, buf, options)

    t1 = time.monotonic()

    for _ in range(args.loops):
        buffer_to_bgr888(fmt, args.width, args.height, bytesperline, buf, options)

    t2 = time.monotonic()
    print(f'Image size: {args.width}x{args.height}, format: {args.format}, stride: {stride}, size {size}, '
          f'{args.loops} loops took {(t2 - t1) * 1000:.3f} ms')

if __name__ == '__main__':
    main()
