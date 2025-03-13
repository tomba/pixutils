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
        rnd = np.random.default_rng(SEED)

        size = fmt.framesize(WIDTH, HEIGHT)

        buf = np.frombuffer(rnd.bytes(size), dtype=np.uint8)
        rgb = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, options)

        with gzip.open(f'{WIDTH}x{HEIGHT}-{fmt}.bin.gz', 'wb') as f:
            f.write(buf.tobytes())

        with gzip.open(f'{WIDTH}x{HEIGHT}-{fmt}-BGR888.bin.gz', 'wb') as f:
            f.write(rgb.tobytes())

if __name__ == '__main__':
    main()
