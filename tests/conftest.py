# SPDX-License-Identifier: BSD-3-Clause

from pixutils.formats import PixelFormat


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, PixelFormat):
        return val.name
    return None
