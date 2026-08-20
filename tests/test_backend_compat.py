#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools

import numpy as np
import pytest
from test_conv import HEIGHT, WIDTH, generate_test_buffer  # type: ignore[import-not-found]

from pixutils.conv import buffer_to_bgr888
from pixutils.formats import PixelColorEncoding, PixelFormat, PixelFormats

BACKENDS = ('opencv', 'pixpat', 'numba', 'numpy')

# For each backend pair we compute the per-pixel absolute BGR difference and
# summarize it with three statistics:
#   * ch_mean — mean absolute diff computed independently per B/G/R channel,
#     then take the max across the three. Using per-channel (not aggregate)
#     means is what makes a channel swap stand out: when R and B are swapped,
#     G matches exactly (~0) while R and B each diverge by ~85 on random
#     uniform data; the aggregate mean would be ~57, which is close to the
#     honest disagreement between different demosaic algorithms, but the
#     per-channel max of ~85 stays well above any legitimate difference.
#   * p99    — 99th percentile of the per-pixel diff. Tolerates rare outliers
#     (chroma samples at a subsampling boundary, clipped saturated pixels).
#   * max    — worst-case single-pixel diff.
#
# Thresholds vary by color class because the backends agree to different
# degrees:
#   * RGB — every backend just reorders bytes, so outputs must be identical.
#   * YUV — all backends share BT.601 limited-range coefficients, but opencv
#     interpolates chroma at subsampling boundaries while numba/numpy do
#     nearest-neighbor, so a handful of LSBs of disagreement is expected.
#   * RAW — opencv, pixpat, numba and numpy each use a different demosaic
#     algorithm, so per-pixel differences are routinely large (just not
#     channel-swap large). On random uniform data the worst pair currently
#     sits at ch_mean 28.1, p99 117, max 198. ch_mean and p99 carry the
#     signal here: a one-pixel channel displacement in one backend's demosaic
#     took them to 48.2 and 134, while max barely moved (196), so the max
#     threshold is only a backstop.
TOLERANCES = {
    'RGB': {'ch_mean': 0.0, 'p99': 0, 'max': 0},
    'YUV': {'ch_mean': 3.0, 'p99': 15, 'max': 25},
    'RAW': {'ch_mean': 32.0, 'p99': 125, 'max': 205},
}


def _category(fmt: PixelFormat) -> str:
    if fmt.color == PixelColorEncoding.RGB:
        return 'RGB'
    if fmt.color == PixelColorEncoding.YUV:
        return 'YUV'
    if fmt.color == PixelColorEncoding.RAW:
        return 'RAW'
    return 'OTHER'


def _format_options(fmt: PixelFormat) -> dict:
    if fmt.color == PixelColorEncoding.YUV and fmt != PixelFormats.Y8:
        # Only range='limited' + encoding='bt601' is accepted by opencv; numba
        # and numpy accept it too, so this enables 3-way comparison where
        # possible and still allows 2-way (numba+numpy) for the rest.
        return {'range': 'limited', 'encoding': 'bt601'}
    return {}


# A small probe size used at test-discovery time to find out which backends
# actually handle a given format. 48 is a multiple of every width alignment
# used by the defined formats (LCM is 12: driven by P030/P230 at 6 and
# XYYY2101010 at 3), and 32 is a multiple of every height alignment (LCM 2).
# The buffer is sized as 48×32 × 8 bytes/pixel × 3 planes — an upper bound
# over every defined format at the probe dimensions; `to_bgr888` slices it
# down to the real framesize.
_PROBE_WIDTH = 48
_PROBE_HEIGHT = 32
_PROBE_BUFFER = np.zeros(_PROBE_WIDTH * _PROBE_HEIGHT * 8 * 3, dtype=np.uint8)


def _probe_backends(fmt: PixelFormat, base_opts: dict) -> list[str]:
    """Return the list of backends that accept this format at the probe size."""
    working = []
    for backend in BACKENDS:
        opts = dict(base_opts) | {'backends': [backend]}
        try:
            buffer_to_bgr888(fmt, _PROBE_WIDTH, _PROBE_HEIGHT, 0, _PROBE_BUFFER, opts)
        except NotImplementedError:
            continue
        working.append(backend)
    return working


def compare_bgr(a: np.ndarray, b: np.ndarray, cat: str) -> str:
    assert a.shape == b.shape and a.dtype == b.dtype == np.uint8
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    if int(diff.max()) == 0:
        return 'identical'
    tol = TOLERANCES[cat]
    ch_mean = diff.mean(axis=(0, 1))
    max_ch_mean = float(ch_mean.max())
    p99 = int(np.percentile(diff, 99))
    mx = int(diff.max())
    msg = f'per-ch-mean={ch_mean.round(2).tolist()} p99={p99} max={mx}'
    assert max_ch_mean <= tol['ch_mean'], (
        f'max-per-ch-mean {max_ch_mean:.2f} > {tol["ch_mean"]}; {msg}'
    )
    assert p99 <= tol['p99'], f'p99 {p99} > {tol["p99"]}; {msg}'
    assert mx <= tol['max'], f'max {mx} > {tol["max"]}; {msg}'
    return f'within tolerance ({msg})'


def _compat_cases():
    """(fmt, b1, b2) for every backend pair that supports the format at the probe size."""
    cases = []
    for fmt in PixelFormats.get_formats():
        if _category(fmt) == 'OTHER':
            continue
        supported = _probe_backends(fmt, _format_options(fmt))
        cases.extend((fmt, b1, b2) for b1, b2 in itertools.combinations(supported, 2))
    return cases


@pytest.mark.parametrize(('fmt', 'b1', 'b2'), _compat_cases())
def test_compat(fmt: PixelFormat, b1: str, b2: str):
    base_opts = _format_options(fmt)
    buf = generate_test_buffer(fmt)
    out1 = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, dict(base_opts) | {'backends': [b1]})
    out2 = buffer_to_bgr888(fmt, WIDTH, HEIGHT, 0, buf, dict(base_opts) | {'backends': [b2]})
    compare_bgr(out1, out2, _category(fmt))
