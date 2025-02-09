#!/usr/bin/python3

# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

import argparse
import gzip
import sys
import typing

import numpy as np
from PyQt6 import QtWidgets

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888
from pixutils.conv.qt import bgr888_to_pix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('width')
    parser.add_argument('height')
    parser.add_argument('format')
    args = parser.parse_args()

    format = PixelFormats.find_by_name(args.format)
    w = int(args.width)
    h = int(args.height)

    if args.file == '-':
        buf = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
    elif args.file.endswith('.gz'):
        with gzip.open(args.file, 'rb') as f:
            data = typing.cast(bytes, f.read())
            buf = np.frombuffer(data, dtype=np.uint8)
    else:
        with open(args.file, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)

    qapp = QtWidgets.QApplication(sys.argv)

    ref = buffer_to_bgr888(format, w, h, 0, buf)

    pix = bgr888_to_pix(ref)

    widget = QtWidgets.QWidget()
    widget.setWindowTitle(format.name)
    layout = QtWidgets.QHBoxLayout()

    label = QtWidgets.QLabel()
    label.setPixmap(pix)
    layout.addWidget(label)

    widget.setLayout(layout)
    widget.show()

    qapp.exec()


if __name__ == '__main__':
    main()
