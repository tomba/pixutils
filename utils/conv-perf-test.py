#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

import numpy as np

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888


def run_one(format_name: str, args: argparse.Namespace, options: dict[str, str | list[str]]):
    fmt = PixelFormats.find_by_name(format_name)

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
    bytesperline = 0 if len(fmt.planes) > 1 else stride

    # Warmup run
    buffer_to_bgr888(fmt, args.width, args.height, bytesperline, buf, options)

    iters = 0
    t_start = time.monotonic()
    while True:
        buffer_to_bgr888(fmt, args.width, args.height, bytesperline, buf, options)
        iters += 1
        elapsed = time.monotonic() - t_start
        if elapsed >= args.time:
            break

    backends_str = args.backends if args.backends else 'default'
    print(
        f'Format: {format_name}, size: {args.width}x{args.height}, backends: {backends_str}, '
        f'stride: {stride}, bufsize: {size}, '
        f'{iters} iters in {elapsed:.3f}s = {iters / elapsed:.1f} iters/s'
    )


def main():
    parser = argparse.ArgumentParser(description='Test conversion performance.')
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')
    parser.add_argument(
        '-f',
        '--format',
        type=str,
        default='XRGB8888',
        help='Pixel format (comma-separated list for multiple formats)',
    )
    parser.add_argument('-t', '--time', type=float, default=1.0, help='Measurement time in seconds')
    parser.add_argument('--stride', type=int, default=0, help='Stride')
    parser.add_argument(
        '--demosaic',
        type=str,
        choices=['3x3', 'bilinear', 'mosaic', 'opencv'],
        default=None,
        help='Demosaic algorithm: 3x3, bilinear, mosaic (no demosaic), or opencv',
    )
    parser.add_argument(
        '--backends',
        type=str,
        default=None,
        help='Comma-separated list of backends in priority order',
    )
    args = parser.parse_args()

    options: dict[str, str | list[str]] = {
        'range': 'limited',
        'encoding': 'bt601',
    }
    if args.demosaic:
        options['demosaic_method'] = args.demosaic
    if args.backends:
        options['backends'] = [b.strip() for b in args.backends.split(',')]

    format_names = [s.strip() for s in args.format.split(',') if s.strip()]

    for format_name in format_names:
        run_one(format_name, args, options)


if __name__ == '__main__':
    main()
