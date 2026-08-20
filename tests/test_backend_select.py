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


RAW_FMT = PixelFormats.SRGGB8


def _raw_buffer():
    return _make_buffer(RAW_FMT)


@pytest.mark.parametrize('method', ['3x3', 'bilinear', 'mosaic', 'opencv'])
def test_pixpat_declines_foreign_demosaic_methods(method):
    # pixpat has one built-in demosaic and no way to select another, so it must
    # bow out rather than silently ignore the request.
    _require_backends('pixpat')
    with pytest.raises(NotImplementedError):
        to_bgr888(
            RAW_FMT,
            WIDTH,
            HEIGHT,
            0,
            _raw_buffer(),
            {'backends': ['pixpat'], 'demosaic_method': method},
        )


def test_pixpat_accepts_its_own_demosaic_method():
    _require_backends('pixpat')
    buf = _raw_buffer()
    implicit = to_bgr888(RAW_FMT, WIDTH, HEIGHT, 0, buf, {'backends': ['pixpat']})
    explicit = to_bgr888(
        RAW_FMT, WIDTH, HEIGHT, 0, buf, {'backends': ['pixpat'], 'demosaic_method': 'pixpat'}
    )
    np.testing.assert_array_equal(explicit, implicit)


def test_demosaic_methods_give_distinct_results():
    # Whatever backends are installed, asking for a specific demosaic method
    # has to reach a backend that implements it.
    _require_backends('numba')
    buf = _raw_buffer()
    outs = [
        to_bgr888(RAW_FMT, WIDTH, HEIGHT, 0, buf, {'demosaic_method': method})
        for method in ('3x3', 'bilinear', 'mosaic')
    ]
    assert not np.array_equal(outs[0], outs[1])
    assert not np.array_equal(outs[0], outs[2])
    assert not np.array_equal(outs[1], outs[2])


def test_unknown_demosaic_method_raises():
    with pytest.raises(ValueError):
        to_bgr888(RAW_FMT, WIDTH, HEIGHT, 0, _raw_buffer(), {'demosaic_method': 'nonsense'})


def test_backend_after_a_declining_numpy_is_tried():
    # The numpy backend has no YVYU converter, opencv does. A backend listed
    # after numpy must still get its turn.
    _require_backends('numpy', 'opencv')
    fmt = PixelFormats.YVYU
    buf = _make_buffer(fmt)

    expected = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['opencv']})
    result = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['numpy', 'opencv']})
    np.testing.assert_array_equal(result, expected)
