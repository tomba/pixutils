# SPDX-License-Identifier: LGPL-3.0-only

from pixutils.formats import PixelFormat


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, PixelFormat):
        return val.name
    return None
