#!/usr/bin/env python3

# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

# Qt methods give false positives about incompatible overrides
# pyright: reportIncompatibleMethodOverride=false

import argparse
import gzip
import sys
import typing

import numpy as np
from PyQt6 import QtCore, QtWidgets

from pixutils.formats import PixelFormats
from pixutils.conv import buffer_to_bgr888
from pixutils.conv.qt import bgr888_to_pix

class ZoomableImageWidget(QtWidgets.QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        self.setPixmap(self.original_pixmap)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            widget_pos = event.position()

            parent = self.parent()
            if parent:
                scroll_area = parent.parent()
                if isinstance(scroll_area, QtWidgets.QScrollArea):
                    viewport_pos = self.mapTo(scroll_area.viewport(), widget_pos.toPoint())
                    mouse_pos = QtCore.QPointF(viewport_pos.x(), viewport_pos.y())
                else:
                    mouse_pos = widget_pos
            else:
                mouse_pos = widget_pos

            if delta > 0:
                self.zoom_in(mouse_pos)
            else:
                self.zoom_out(mouse_pos)
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom_in(self, center_point=None):
        old_scale = self.scale_factor
        self.scale_factor *= 1.1
        self.update_pixmap(center_point, old_scale)

    def zoom_out(self, center_point=None):
        old_scale = self.scale_factor
        self.scale_factor /= 1.1
        if self.scale_factor < 0.1:
            self.scale_factor = 0.1
        self.update_pixmap(center_point, old_scale)

    def reset_zoom(self):
        self.scale_factor = 1.0
        self.update_pixmap()

    def update_pixmap(self, center_point=None, old_scale=None):
        if self.scale_factor == 1.0:
            scaled_pixmap = self.original_pixmap
        else:
            size = self.original_pixmap.size() * self.scale_factor
            scaled_pixmap = self.original_pixmap.scaled(
                size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation
            )
        self.setPixmap(scaled_pixmap)
        self.resize(scaled_pixmap.size())

        if center_point and old_scale:
            parent = self.parent()
            if parent:
                scroll_area = parent.parent()
                if isinstance(scroll_area, QtWidgets.QScrollArea):
                    scale_ratio = self.scale_factor / old_scale

                    h_bar = scroll_area.horizontalScrollBar()
                    v_bar = scroll_area.verticalScrollBar()

                    if h_bar and v_bar:
                        old_scroll_x = h_bar.value()
                        old_scroll_y = v_bar.value()

                        mouse_x = center_point.x()
                        mouse_y = center_point.y()

                        new_scroll_x = (old_scroll_x + mouse_x) * scale_ratio - mouse_x
                        new_scroll_y = (old_scroll_y + mouse_y) * scale_ratio - mouse_y

                        h_bar.setValue(int(new_scroll_x))
                        v_bar.setValue(int(new_scroll_y))

class ImageViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, pixmap, title='Image Viewer'):
        super().__init__()
        self.image_widget = ZoomableImageWidget(pixmap)
        self.init_ui()
        self.setWindowTitle(title)

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout()

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(self.image_widget)
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(scroll_area)
        central_widget.setLayout(layout)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_R:
            self.image_widget.reset_zoom()
        else:
            super().keyPressEvent(event)

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

    window = ImageViewerWindow(pix, f'{format.name} - Ctrl+Wheel to zoom, R to reset')
    window.resize(800, 600)
    window.show()

    qapp.exec()


if __name__ == '__main__':
    main()
