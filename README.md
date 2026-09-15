[![Lint Status](https://github.com/tomba/pixutils/actions/workflows/ci.yml/badge.svg)](https://github.com/tomba/pixutils/actions/workflows/ci.yml)

# Linux Pix Utils

Pixel format definitions and conversions for Linux V4L2 and DRM/KMS
buffers, and command line tools built on them.

## Installation

The package is not on PyPI. Install it from git:

    pip install 'pixutils @ git+https://github.com/tomba/pixutils.git'

The base package has no dependencies. It provides the pixel format tables
(`pixutils.formats`) and the dma-buf and dma-heap helpers. The rest is
behind extras:

| Extra    | Provides                                                    |
| -------- | ----------------------------------------------------------- |
| `conv`   | `pixutils.conv`, with numpy and opencv-python               |
| `pixpat` | The pixpat conversion backend for RAW formats               |
| `qt`     | `pixutils.qt`, the viewer widget, with PyQt6 and pyqtgraph  |
| `png`    | Pillow, used by `pixutils-conv-png`                         |
| `tools`  | `conv`, `png` and `qt`: everything the command line tools need |
| `all`    | Everything above, plus pytest                                |

Extras are given in the usual way:

    pip install 'pixutils[conv] @ git+https://github.com/tomba/pixutils.git'

## Tools

- `pixutils-view`: view a raw framebuffer file in a Qt window.
- `pixutils-conv-png`: convert a raw framebuffer file to PNG.

To install only the tools, use pipx or `uv tool`:

    pipx install 'pixutils[tools] @ git+https://github.com/tomba/pixutils.git'

Both tools take the file name, followed by the width, height and pixel
format. The parameters are parsed from the file name when omitted, for
example `frame-640x480-NV12.raw`. A `.gz` file is decompressed.

The numpy backend covers the common formats on its own; opencv-python
and pixpat add speed and a few formats and demosaic methods. On a
system where they or Qt cannot be installed, install the base package
and add the modules the tool needs by hand:

    pipx install 'pixutils @ git+https://github.com/tomba/pixutils.git'
    pipx inject pixutils numpy Pillow

## License

This project is covered by the [LGPL-3.0](LICENSE.md) license.
