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


def _require_backend(name):
    from pixutils.conv import get_backends

    if not get_backends([name]):
        pytest.skip(f'{name} backend unavailable')


@pytest.mark.parametrize(
    'fmt',
    [
        PixelFormats.RGB888,
        PixelFormats.YUYV,
        PixelFormats.NV12,
        PixelFormats.NV16,
        PixelFormats.YUV420,
        PixelFormats.YUV422,
        PixelFormats.SRGGB10P,
    ],
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


def test_frame_equality_is_identity():
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    f1 = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)
    f2 = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)

    # The planes are ndarrays, so field-wise comparison cannot work. Frame
    # therefore uses identity semantics rather than raising.
    alias = f1
    assert f1 == alias
    assert f1 != f2


def test_frame_is_hashable():
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    f1 = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)
    f2 = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, buf)

    assert len({f1, f2, f1}) == 2


@pytest.mark.parametrize(
    'fmt',
    [
        PixelFormats.RGB888,
        PixelFormats.YUYV,
        PixelFormats.Y8,
        PixelFormats.NV12,
        PixelFormats.NV16,
        PixelFormats.YUV420,
        PixelFormats.YUV422,
        PixelFormats.SRGGB8,
    ],
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
    [
        PixelFormats.RGB888,
        PixelFormats.YUYV,
        PixelFormats.NV12,
        PixelFormats.NV16,
        PixelFormats.YUV420,
        PixelFormats.YUV422,
    ],
)
def test_from_planes_matches_from_single_buffer(fmt):
    _require_backend('numpy')
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


@pytest.mark.parametrize(
    'fmt',
    [PixelFormats.NV12, PixelFormats.NV16, PixelFormats.YUV420, PixelFormats.YUV422],
)
def test_to_bgr888_accepts_sequence_arr(fmt):
    _require_backend('numpy')
    buf = _make_buffer(fmt)
    opts = {'backends': ['numpy']}
    single = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts)
    seq = to_bgr888(fmt, WIDTH, HEIGHT, 0, _split_planes(fmt, buf), opts)
    np.testing.assert_array_equal(seq, single)


@pytest.mark.parametrize(
    'fmt',
    [PixelFormats.NV12, PixelFormats.NV16, PixelFormats.YUV420, PixelFormats.YUV422],
)
def test_buffer_to_bgr888_accepts_sequence(fmt):
    from pixutils.conv import buffer_to_bgr888

    _require_backend('numpy')
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
        (PixelFormats.RGB888, 'opencv'),
        (PixelFormats.YUYV, 'numpy'),
        (PixelFormats.YUYV, 'numba'),
        (PixelFormats.YUYV, 'opencv'),
        (PixelFormats.NV12, 'numpy'),
        (PixelFormats.NV12, 'numba'),
        (PixelFormats.NV16, 'numpy'),
        (PixelFormats.NV16, 'numba'),
        (PixelFormats.YUV420, 'numpy'),
        (PixelFormats.YUV422, 'numpy'),
    ],
)
def test_crop_matches_full_subregion(fmt, backend):
    _require_backend(backend)
    x, y, w, h = CROP
    buf = _make_buffer(fmt)
    opts = {'backends': [backend]}

    full = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts)
    cropped = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts, crop=CROP)
    np.testing.assert_array_equal(cropped, full[y : y + h, x : x + w])


def test_crop_misaligned_raises():
    fmt = PixelFormats.NV12
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, _make_buffer(fmt))
    with pytest.raises(ValueError):
        frame.crop(1, 0, 8, 4)  # odd x, NV12 requires multiples of 2


@pytest.mark.parametrize(
    ('fmt', 'rect'),
    [
        # Y210 packs two pixels into one 8-byte block, but declares
        # pixel_align (1, 1), so an odd x or w has no byte representation.
        (PixelFormats.Y210, (1, 0, 8, 4)),
        (PixelFormats.Y210, (0, 0, 7, 4)),
        # YUV420 declares pixel_align (1, 1) as well, but its chroma planes
        # are subsampled by 2 in both directions.
        (PixelFormats.YUV420, (1, 0, 8, 4)),
        (PixelFormats.YUV420, (0, 1, 8, 4)),
        (PixelFormats.YUV420, (0, 0, 7, 4)),
        (PixelFormats.YUV420, (0, 0, 8, 3)),
        # YUV422 subsamples chroma horizontally only.
        (PixelFormats.YUV422, (1, 0, 8, 4)),
        (PixelFormats.YUV422, (0, 0, 7, 4)),
    ],
)
def test_crop_macropixel_misaligned_raises(fmt, rect):
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, _make_buffer(fmt))
    with pytest.raises(ValueError):
        frame.crop(*rect)


def test_crop_out_of_bounds_raises():
    fmt = PixelFormats.RGB888
    frame = Frame.from_single_buffer(fmt, WIDTH, HEIGHT, 0, _make_buffer(fmt))
    with pytest.raises(ValueError):
        frame.crop(0, 0, WIDTH + 2, HEIGHT)


def test_opencv_bows_out_on_cropped_nv12():
    _require_backend('opencv')
    fmt = PixelFormats.NV12
    buf = _make_buffer(fmt)
    # opencv-only on a cropped multi-plane frame should have no usable backend
    with pytest.raises(NotImplementedError):
        to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['opencv']}, crop=CROP)
    # but opencv handles the un-cropped frame fine
    to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, {'backends': ['opencv']})


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
@pytest.mark.parametrize('fmt', [PixelFormats.SRGGB8, PixelFormats.SRGGB10P])
def test_raw_crop_matches_full_interior(fmt, backend):
    _require_backend(backend)
    x, y, w, h = CROP
    buf = _make_buffer(fmt)
    opts = {'backends': [backend]}

    full = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts)
    cropped = to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, opts, crop=CROP)

    # The 3x3 demosaic zero-pads at the crop border, so only the interior
    # (excluding the 1-px border) is expected to match the full-frame result.
    np.testing.assert_array_equal(cropped[1:-1, 1:-1], full[y + 1 : y + h - 1, x + 1 : x + w - 1])
