#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888


def run_one(
    format_name: str, args: argparse.Namespace, options: dict[str, str | list[str]]
) -> dict:
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

    rng = np.random.default_rng(0)
    buf = rng.integers(0, 256, size=size, dtype=np.uint8)

    stride = args.stride if args.stride > 0 else fmt.stride(args.width, 0)
    bytesperline = 0 if len(fmt.planes) > 1 else stride

    # Warmup
    for _ in range(3):
        buffer_to_bgr888(fmt, args.width, args.height, bytesperline, buf, options)

    min_iter_s = float('inf')
    gc.disable()
    try:
        iters = 0
        t_start = time.perf_counter()
        t_prev = t_start
        while True:
            buffer_to_bgr888(fmt, args.width, args.height, bytesperline, buf, options)
            iters += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt < min_iter_s:
                min_iter_s = dt
            t_prev = t_now
            elapsed = t_now - t_start
            if elapsed >= args.time:
                break
    finally:
        gc.enable()

    mean_iters_per_s = iters / elapsed
    peak_iters_per_s = 1.0 / min_iter_s

    backends_str = args.backends if args.backends else 'default'
    print(
        f'{format_name} {args.width}x{args.height}, backends: {backends_str}, '
        f'stride: {stride} '
        f'{iters} iters in {elapsed:.3f}s = {mean_iters_per_s:.1f}/s mean, '
        f'{peak_iters_per_s:.1f}/s peak'
    )

    return {
        'format': format_name,
        'strides': list([stride]),
        'bufsize': size,
        'iters': iters,
        'elapsed': elapsed,
        'iters_per_s': mean_iters_per_s,
        'min_iter_s': min_iter_s,
    }


def git_info() -> tuple[str | None, bool, str | None]:
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        subject = subprocess.run(
            ['git', 'show', '-s', '--format=%s', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, dirty, subject
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False, None


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
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Write results as JSON to this file (for conv-perf-compare.py)',
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

    runs = []
    for format_name in format_names:
        runs.append(run_one(format_name, args, options))

    if args.output:
        commit, dirty, subject = git_info()
        meta = {
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'git_commit': commit,
            'git_dirty': dirty,
            'git_subject': subject,
            'hostname': socket.gethostname(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'width': args.width,
            'height': args.height,
            'padding': 0,
            'backends': options.get('backends', ['default']),
            'options': {k: v for k, v in options.items() if k != 'backends'},
            'measurement_time': args.time,
        }
        with open(args.output, 'w') as f:
            json.dump({'meta': meta, 'runs': runs}, f, indent=2)


if __name__ == '__main__':
    main()
