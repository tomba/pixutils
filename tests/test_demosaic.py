# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the RAW demosaic algorithms."""

import numpy as np
import pytest

from pixutils.conv import get_backends, to_bgr888
from pixutils.conv.raw import BayerPattern
from pixutils.formats import PixelFormats

WIDTH = 16
HEIGHT = 8
VALUE = 200

RAW8_FORMATS = [
    PixelFormats.SBGGR8,
    PixelFormats.SGBRG8,
    PixelFormats.SGRBG8,
    PixelFormats.SRGGB8,
]


def _require_backend(name):
    if not get_backends([name]):
        pytest.skip(f'{name} backend unavailable')


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize(('channel', 'corner'), [(0, 'r0'), (2, 'b0')], ids=['red', 'blue'])
@pytest.mark.parametrize('fmt', RAW8_FORMATS)
def test_3x3_demosaic_keeps_samples_in_place(fmt, channel, corner, backend):
    """A lone Bayer sample must come out at its own pixel, not next to it.

    A 3x3 window centred on a red (or blue) sample covers exactly one red (or
    blue) Bayer position, so the weighted average there is the sample value
    itself -- and every other pixel averages it with zeros, so the sample
    position is also the unique maximum of that channel.
    """
    _require_backend(backend)

    assert fmt.bayer_pattern is not None
    x, y = getattr(BayerPattern.from_pattern(fmt.bayer_pattern), corner)
    # Move off the border so the whole 3x3 window is inside the image.
    x, y = x + 2, y + 2

    data = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    data[y, x] = VALUE

    opts = {'backends': [backend], 'demosaic_method': '3x3'}
    out = to_bgr888(fmt, WIDTH, HEIGHT, 0, data.ravel(), opts)

    plane = out[..., channel]
    assert plane[y, x] == VALUE
    assert np.unravel_index(np.argmax(plane), plane.shape) == (y, x)
    assert np.count_nonzero(plane == VALUE) == 1
