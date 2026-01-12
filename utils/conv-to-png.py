#!/usr/bin/env python3

import argparse
import gzip
import os
import re
import sys
import typing

import numpy as np
from PIL import Image

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888


def parse_filename_heuristics(filename):
    """Try to parse width, height, format, range and encoding from filename"""
    basename = os.path.basename(filename)
    if basename.endswith('.gz'):
        basename = os.path.splitext(basename)[0]

    basename_no_ext = os.path.splitext(basename)[0]

    parts = re.split(r'[-_]', basename_no_ext)
    parts = [p for p in parts if p]

    if not parts:
        return None

    encoding_options = {'bt601', 'bt709', 'bt2020'}
    range_options = {'full', 'limited'}

    result = {}

    remaining_parts = parts[:]

    for part in parts:
        part_lower = part.lower()
        if part_lower in encoding_options:
            result['encoding'] = part_lower
            remaining_parts.remove(part)
        elif part_lower in range_options:
            result['range'] = part_lower
            remaining_parts.remove(part)

    for part in remaining_parts[:]:
        try:
            result['format'] = PixelFormats.find_by_name(part)
            remaining_parts.remove(part)
            break
        except StopIteration:
            continue

    if 'format' not in result:
        return None

    for part in remaining_parts:
        if 'x' in part:
            try:
                w, h = part.split('x', 1)
                result['width'] = int(w)
                result['height'] = int(h)
                break
            except (ValueError, IndexError):
                continue

    if 'width' not in result or 'height' not in result:
        return None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument(
        'width', nargs='?', help='Width in pixels (optional if filename contains heuristics)'
    )
    parser.add_argument(
        'height', nargs='?', help='Height in pixels (optional if filename contains heuristics)'
    )
    parser.add_argument(
        'format', nargs='?', help='Pixel format (optional if filename contains heuristics)'
    )
    parser.add_argument('--range', choices=['full', 'limited'], help='Color range')
    parser.add_argument('--encoding', choices=['bt601', 'bt709', 'bt2020'], help='Color encoding')
    parser.add_argument('--demosaic', choices=['3x3', 'bilinear', 'mosaic'], help='Demosaic method')
    parser.add_argument('--output', '-o', required=True, help='Output PNG filename')
    args = parser.parse_args()

    if args.width and args.height and args.format:
        format = PixelFormats.find_by_name(args.format)
        w = int(args.width)
        h = int(args.height)
        detected_encoding = args.encoding
        detected_range = args.range
    else:
        parsed = parse_filename_heuristics(args.file)
        if parsed is None:
            if not args.width or not args.height or not args.format:
                parser.error(
                    'Could not detect parameters from filename. Please provide width, height, and format arguments.'
                )
            format = PixelFormats.find_by_name(args.format)
            w = int(args.width)
            h = int(args.height)
            detected_encoding = args.encoding
            detected_range = args.range
        else:
            format = parsed['format']
            w = parsed['width']
            h = parsed['height']
            detected_encoding = parsed.get('encoding')
            detected_range = parsed.get('range')

    if args.file == '-':
        buf = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
    elif args.file.endswith('.gz'):
        with gzip.open(args.file, 'rb') as f:
            data = typing.cast(bytes, f.read())
            buf = np.frombuffer(data, dtype=np.uint8)
    else:
        with open(args.file, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)

    options = {}
    final_range = args.range if args.range else detected_range
    final_encoding = args.encoding if args.encoding else detected_encoding

    if final_range:
        options['range'] = final_range
    if final_encoding:
        options['encoding'] = final_encoding
    if args.demosaic:
        options['demosaic_method'] = args.demosaic

    bgr = buffer_to_bgr888(format, w, h, 0, buf, options)

    img = Image.fromarray(bgr)
    img.save(args.output)


if __name__ == '__main__':
    main()
