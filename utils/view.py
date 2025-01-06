#!/usr/bin/python3

# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

import os
import gzip
import sys

import numpy as np
from PyQt6 import QtWidgets

from pixutils.formats import PixelFormats, str_to_fourcc
from pixutils.conv import buffer_to_bgr888
from pixutils.conv.qt import bgr888_to_pix

TEST_PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    qapp = QtWidgets.QApplication(sys.argv)

    # Some formats are not supported yet

    fmts = [
        #'BG16',
        'BG24',
        #'BX24',
        'NV12',
        #'NV21',
        #'RG16',
        'RG24',
        #'RX24',
        'UYVY',
        'XB24',
        'XR24',
        'YUYV',
    ]

    #fmts = [ 'YUYV' ]

    w = 640
    h = 480

    fname = f'{TEST_PATH}/data/test-{w}-{h}-BG24.bin.gz'

    with gzip.open(fname, 'rb') as f:
        ref_buf = np.frombuffer(f.read(), dtype=np.uint8)
    ref = buffer_to_bgr888(PixelFormats.BGR888, w, h, 0, ref_buf)
    ref = ref.astype(np.int16)

    for fourccstr in fmts:
        try:
            fmt = PixelFormats.find_drm_fourcc(str_to_fourcc(fourccstr))
        except StopIteration:
            print(f'fourcc {fourccstr} not supported')
            continue

        print(f'Showing {fmt} ({fourccstr})')

        bytesperline = 0 #fmt.stride(w)

        fname = f'{TEST_PATH}/data/test-{w}-{h}-{fourccstr}.bin.gz'

        with gzip.open(fname, 'rb') as f:
            data_in = np.frombuffer(f.read(), dtype=np.uint8)

        # Note: the yuv test images are in bt601 limited
        options = {
            'range': 'limited',
            'encoding': 'bt601',
        }

        rgb = buffer_to_bgr888(fmt, w, h, bytesperline, data_in, options)

        assert rgb.shape == ref.shape

        rgb = rgb.astype(np.int16)

        diff = rgb - ref

        if not diff.any():
            print('  Match')
            continue

        diff = abs(diff)

        b = diff[:,:,0]
        g = diff[:,:,1]
        r = diff[:,:,2]

        print('  min()  {:5} {:5} {:5}'.format(b.min(), g.min(), r.min()))
        print('  max()  {:5} {:5} {:5}'.format(b.max(), g.max(), r.max()))
        print('  mean() {:5.2} {:5.2} {:5.2}'.format(b.mean(), g.mean(), r.mean()))

        widget = QtWidgets.QWidget()
        widget.setWindowTitle(fourccstr)
        layout = QtWidgets.QHBoxLayout()

        label = QtWidgets.QLabel()
        label.setPixmap(bgr888_to_pix(ref.astype(np.uint8)))
        layout.addWidget(label)

        label = QtWidgets.QLabel()
        label.setPixmap(bgr888_to_pix(rgb.astype(np.uint8)))
        layout.addWidget(label)

        label = QtWidgets.QLabel()
        label.setPixmap(bgr888_to_pix(diff.astype(np.uint8)))
        layout.addWidget(label)

        widget.setLayout(layout)
        widget.showMaximized()

        qapp.exec()


if __name__ == '__main__':
    main()
