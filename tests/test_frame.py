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


CROP = (4, 2, 8, 4)  # (x, y, w, h), aligned to 2x2


@pytest.mark.parametrize(
    ('fmt', 'backend'),
    [
        (PixelFormats.RGB888, 'numpy'),
        (PixelFormats.YUYV, 'numpy'),
        (PixelFormats.NV12, 'numpy'),
        (PixelFormats.NV12, 'numba'),
        (PixelFormats.YUV420, 'numpy'),
    ],
)
def test_crop_matches_full_subregion(fmt, backend):
    x, y, w, h = CROP
    buf = _make_buffer(fmt)
    opts = {'backends': [backend]}

    try:
        full = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts)
    except NotImplementedError:
        pytest.skip(f'{backend} unavailable')

    cropped = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts, crop=CROP)
    np.testing.assert_array_equal(cropped, full[y : y + h, x : x + w])


def test_crop_misaligned_raises():
    fmt = PixelFormats.NV12
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, _make_buffer(fmt))
    with pytest.raises(ValueError):
        frame.crop(1, 0, 8, 4)  # odd x, NV12 requires multiples of 2


def test_crop_out_of_bounds_raises():
    fmt = PixelFormats.RGB888
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, _make_buffer(fmt))
    with pytest.raises(ValueError):
        frame.crop(0, 0, WIDTH + 2, HEIGHT)


def test_opencv_bows_out_on_cropped_nv12():
    cv2 = pytest.importorskip('cv2')  # noqa: F841
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    # opencv-only on a cropped multi-plane frame should have no usable backend
    with pytest.raises(NotImplementedError):
        to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['opencv']}, crop=CROP)
    # but opencv handles the un-cropped frame fine
    to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['opencv']})


@pytest.mark.parametrize('fmt', [PixelFormats.SRGGB8, PixelFormats.SRGGB10P])
def test_raw_crop_matches_full_interior(fmt):
    x, y, w, h = CROP
    buf = _make_buffer(fmt)
    opts = {'backends': ['numpy']}

    full = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts)
    cropped = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts, crop=CROP)

    # The 3x3 demosaic zero-pads at the crop border, so only the interior
    # (excluding the 1-px border) is expected to match the full-frame result.
    np.testing.assert_array_equal(cropped[1:-1, 1:-1], full[y + 1 : y + h - 1, x + 1 : x + w - 1])
