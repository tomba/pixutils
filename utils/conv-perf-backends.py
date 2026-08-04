#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np

from pixutils.conv import buffer_to_bgr888
from pixutils.conv.backends import get_backends
from pixutils.formats import PixelFormats


def all_format_names() -> list[str]:
    return sorted(f.name for f in PixelFormats.get_formats())


def run_one(
    fmt,
    width: int,
    height: int,
    strides: tuple[int, ...],
    buf: np.ndarray,
    backend: str,
    base_options: dict,
    measure_time: float,
) -> tuple[float, float] | str | None:
    options = dict(base_options)
    options['backends'] = [backend]

    # Warmup. NotImplementedError means the backend doesn't support this format
    # (single-element backends list, so no fallback). Catch broader exceptions
    # too: when sweeping every format some (format, backend, width) triples hit
    # format-specific errors (e.g. pixpat alignment), and we don't want a
    # single bad cell to abort the whole matrix.
    try:
        for _ in range(3):
            buffer_to_bgr888(fmt, width, height, strides, buf, options)
    except NotImplementedError:
        return None
    except Exception as e:  # noqa: BLE001 - see comment above
        return f'{type(e).__name__}: {e}'

    min_iter_s = float('inf')
    gc.disable()
    try:
        iters = 0
        t_start = time.perf_counter()
        t_prev = t_start
        while True:
            buffer_to_bgr888(fmt, width, height, strides, buf, options)
            iters += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev
            min_iter_s = min(min_iter_s, dt)
            t_prev = t_now
            elapsed = t_now - t_start
            if elapsed >= measure_time:
                break
    finally:
        gc.enable()

    return iters / elapsed, 1.0 / min_iter_s


def print_table(rows: list[list[str]], aligns: list[str]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r_idx, row in enumerate(rows):
        parts = []
        for i, cell in enumerate(row):
            if aligns[i] == 'l':
                parts.append(cell.ljust(widths[i]))
            else:
                parts.append(cell.rjust(widths[i]))
        print('  '.join(parts).rstrip())
        if r_idx == 0:
            print('  '.join('-' * w for w in widths))


def format_options(options: dict) -> str:
    if not options:
        return '{}'
    return '{' + ', '.join(f'{k}:{v}' for k, v in options.items()) + '}'


def main():
    parser = argparse.ArgumentParser(
        description='Compare conversion performance across multiple backends.'
    )
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')
    parser.add_argument(
        '-f',
        '--format',
        type=str,
        default=None,
        help='Pixel format (comma-separated list). Default: every known format '
        '(rows with no supporting backend are dropped from the table).',
    )
    parser.add_argument(
        '-t', '--time', type=float, default=1.0, help='Measurement time in seconds per cell'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=0,
        help="Extra bytes added to each plane's natural stride",
    )
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
        default='opencv,pixpat,numba,numpy',
        help='Comma-separated list of backends to compare (default: all four)',
    )
    args = parser.parse_args()

    requested = [b.strip() for b in args.backends.split(',') if b.strip()]
    backends = get_backends(requested)
    if not backends:
        sys.exit(f'No available backends in requested set: {requested}')

    base_options: dict = {
        'range': 'limited',
        'encoding': 'bt601',
    }
    if args.demosaic:
        base_options['demosaic_method'] = args.demosaic

    if args.format:
        format_names = [s.strip() for s in args.format.split(',') if s.strip()]
    else:
        format_names = all_format_names()

    print(
        f'Config: {args.width}x{args.height}  padding={args.padding}  '
        f'options={format_options(base_options)}  backends={",".join(backends)}'
    )
    print()

    # results[format_name][backend] = (mean, peak) | error-message | None
    results: dict[str, dict[str, tuple[float, float] | str | None]] = {}
    for format_name in format_names:
        fmt = PixelFormats.find_by_name(format_name)
        strides = tuple(fmt.stride(args.width, i) + args.padding for i in range(len(fmt.planes)))
        size = sum(fmt.planesize(strides[i], args.height, i) for i in range(len(fmt.planes)))

        rng = np.random.default_rng(0)
        buf = rng.integers(0, 256, size=size, dtype=np.uint8)

        results[format_name] = {}
        for backend in backends:
            r = run_one(
                fmt, args.width, args.height, strides, buf, backend, base_options, args.time
            )
            results[format_name][backend] = r
            if r is None:
                print(f'  {format_name} / {backend}: unsupported')
            elif isinstance(r, str):
                print(f'  {format_name} / {backend}: error ({r})')
            else:
                print(f'  {format_name} / {backend}: {r[0]:.1f}/s mean, {r[1]:.1f}/s peak')

    print()

    header = ['Format']
    aligns = ['l']
    for backend in backends:
        header.extend([f'{backend} mean', f'{backend} peak'])
        aligns.extend(['r', 'r'])

    rows: list[list[str]] = [header]
    for format_name in format_names:
        cells = [results[format_name][b] for b in backends]
        if all(not isinstance(c, tuple) for c in cells):
            continue
        row = [format_name]
        for r in cells:
            if isinstance(r, tuple):
                row.extend([f'{r[0]:.1f}/s', f'{r[1]:.1f}/s'])
            elif isinstance(r, str):
                row.extend(['err', 'err'])
            else:
                row.extend(['—', '—'])
        rows.append(row)

    print_table(rows, aligns)


if __name__ == '__main__':
    main()
