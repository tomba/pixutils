#!/usr/bin/env python3

import gzip
import numpy as np

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888

WIDTH = 640
HEIGHT = 480

SEED = 1234

FMTS = [
    PixelFormats.BGR888,
    PixelFormats.RGB888,
    PixelFormats.XBGR8888,
    PixelFormats.XRGB8888,

    PixelFormats.NV12,
    PixelFormats.UYVY,
    PixelFormats.YUYV,

    PixelFormats.SRGGB10,
    PixelFormats.SRGGB10P,
    PixelFormats.SRGGB12,
    PixelFormats.SRGGB16,
    PixelFormats.SRGGB8,
]

def main():
    options = {
        'range': 'limited',
        'encoding': 'bt601',
    }

    for fmt in FMTS:
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

        rgb = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, options)

        src_file = f'{WIDTH}x{HEIGHT}-{fmt}.bin.gz'
        rgb_file = f'{WIDTH}x{HEIGHT}-{fmt}-BGR888.bin.gz'

        with open(src_file, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
                gz.write(buf.tobytes())

        with open(rgb_file, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
                gz.write(rgb.tobytes())

if __name__ == '__main__':
    main()
