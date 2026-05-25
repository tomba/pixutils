# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from pixutils.conv import frame_to_bgr888, to_bgr888
from pixutils.conv.frame import Frame
from pixutils.formats import PixelFormats

WIDTH = 16
HEIGHT = 8


def _make_buffer(fmt):
    size = fmt.framesize(WIDTH, HEIGHT)
    return np.arange(size, dtype=np.uint8)


@pytest.mark.parametrize(
    'fmt',
    [PixelFormats.RGB888, PixelFormats.YUYV, PixelFormats.NV12, PixelFormats.YUV420],
)
def test_natural_strides_split(fmt):
    buf = _make_buffer(fmt)
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)

    n = len(fmt.planes)
    assert len(frame.planes) == n
    assert len(frame.strides) == n
    assert frame.strides == tuple(fmt.stride(WIDTH, i) for i in range(n))

    # Each plane is the correct slice of the input, in order, and is a view.
    offset = 0
    for i, plane in enumerate(frame.planes):
        size = fmt.planesize(frame.strides[i], HEIGHT, i)
        assert plane.size == size
        np.testing.assert_array_equal(plane, buf[offset : offset + size])
        assert np.shares_memory(plane, buf)
        offset += size
    assert offset == fmt.framesize(WIDTH, HEIGHT)


def test_combined_is_zero_copy_for_single_buffer():
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)

    combined = frame.combined()
    np.testing.assert_array_equal(combined, buf)
    assert np.shares_memory(combined, buf)


def test_single_plane_combined_is_the_plane():
    fmt = PixelFormats.RGB888
    buf = _make_buffer(fmt)
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)

    assert len(frame.planes) == 1
    assert np.shares_memory(frame.combined(), buf)


def test_padded_stride_single_int():
    fmt = PixelFormats.NV12
    pad = 8
    y_stride = fmt.stride(WIDTH, 0) + pad
    # plane sizes for the padded plane-0 stride, extrapolated for plane 1
    strides = tuple(fmt.extrapolate_stride(y_stride, i) for i in range(len(fmt.planes)))
    size = sum(fmt.planesize(strides[i], HEIGHT, i) for i in range(len(fmt.planes)))
    buf = np.arange(size, dtype=np.uint8)

    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, y_stride, buf)
    assert frame.strides == strides
    assert frame.strides[0] == y_stride


def test_too_small_buffer_raises():
    fmt = PixelFormats.RGB888
    buf = np.zeros(8, dtype=np.uint8)
    with pytest.raises(ValueError):
        Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)


def test_wrong_strides_sequence_length_raises():
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    with pytest.raises(ValueError):
        Frame.from_single_buffer(fmt, WIDTH, HEIGHT, [fmt.stride(WIDTH, 0)], buf)


def test_zero_stride_in_sequence_raises():
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    with pytest.raises(ValueError):
        Frame.from_single_buffer(fmt, WIDTH, HEIGHT, [fmt.stride(WIDTH, 0), 0], buf)


@pytest.mark.parametrize(
    'fmt',
    [PixelFormats.RGB888, PixelFormats.YUYV, PixelFormats.NV12, PixelFormats.YUV420],
)
def test_frame_to_bgr888_matches_to_bgr888(fmt):
    buf = _make_buffer(fmt)
    try:
        expected = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf)
    except NotImplementedError:
        pytest.skip('No backend available')

    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)
    np.testing.assert_array_equal(frame_to_bgr888(frame), expected)


def _split_planes(fmt, buf):
    """Split a single concatenated buffer into independent per-plane copies."""
    arr = np.frombuffer(buf, np.uint8)
    out = []
    offset = 0
    for i in range(len(fmt.planes)):
        size = fmt.planesize(fmt.stride(WIDTH, i), HEIGHT, i)
        out.append(np.array(arr[offset : offset + size]))  # independent copy
        offset += size
    return out


@pytest.mark.parametrize(
    'fmt',
    [PixelFormats.RGB888, PixelFormats.YUYV, PixelFormats.NV12, PixelFormats.YUV420],
)
def test_from_planes_matches_from_single_buffer(fmt):
    buf = _make_buffer(fmt)
    opts = {'backends': ['numpy']}  # same backend on both sides

    single = frame_to_bgr888(Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf), opts)
    planes = frame_to_bgr888(Frame.from_planes(fmt, WIDTH, HEIGHT, _split_planes(fmt, buf)), opts)
    np.testing.assert_array_equal(planes, single)


def test_from_planes_wrong_count_raises():
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    with pytest.raises(ValueError):
        Frame.from_planes(fmt, WIDTH, HEIGHT, _split_planes(fmt, buf)[:1])


@pytest.mark.parametrize('fmt', [PixelFormats.NV12, PixelFormats.YUV420])
def test_to_bgr888_accepts_sequence_arr(fmt):
    buf = _make_buffer(fmt)
    opts = {'backends': ['numpy']}
    single = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts)
    seq = to_bgr888(fmt, WIDTH, HEIGHT, 0, _split_planes(fmt, buf), opts)
    np.testing.assert_array_equal(seq, single)


@pytest.mark.parametrize('fmt', [PixelFormats.NV12, PixelFormats.YUV420])
def test_buffer_to_bgr888_accepts_sequence(fmt):
    from pixutils.conv import buffer_to_bgr888

    buf = _make_buffer(fmt)
    opts = {'backends': ['numpy']}
    single = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, buf.tobytes(), opts)
    planes = [p.tobytes() for p in _split_planes(fmt, buf)]
    seq = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, planes, opts)
    np.testing.assert_array_equal(seq, single)
