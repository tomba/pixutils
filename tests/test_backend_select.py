# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how frame_to_bgr888() picks a backend from options['backends']."""

import numpy as np
import pytest

from pixutils.conv import get_backends, to_bgr888
from pixutils.formats import PixelFormats

WIDTH = 16
HEIGHT = 8


def _require_backends(*names):
    for name in names:
        if not get_backends([name]):
            pytest.skip(f'{name} backend unavailable')


def _make_buffer(fmt):
    return np.arange(fmt.framesize(WIDTH, HEIGHT), dtype=np.uint8)


def test_backend_after_a_declining_numpy_is_tried():
    # The numpy backend has no YVYU converter, opencv does. A backend listed
    # after numpy must still get its turn.
    _require_backends('numpy', 'opencv')
    fmt = PixelFormats.YVYU
    buf = _make_buffer(fmt)

    expected = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['opencv']})
    result = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['numpy', 'opencv']})
    np.testing.assert_array_equal(result, expected)
