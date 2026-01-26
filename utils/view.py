#!/usr/bin/env python3

# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

# Qt methods give false positives about incompatible overrides
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import typing

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

from pixutils.formats import PixelFormats, fourcc_to_str


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = QtCore.pyqtSignal(float)
    mousePositionChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, pixmap: QtGui.QPixmap) -> None:
        super().__init__()

        # Create scene and add pixmap
        self._scene = QtWidgets.QGraphicsScene()
        self.pixmap_item = self._scene.addPixmap(pixmap)
        self.setScene(self._scene)

        # Configure view
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)

        # Track mouse position for zoom-under-cursor
        self.target_viewport_pos = QtCore.QPointF()
        self.target_scene_pos = QtCore.QPointF()
        self.setMouseTracking(True)

        # Base for exponential zoom calculation: raised to wheel angle delta to get smooth,
        # proportional zoom (larger wheel movements = larger zoom changes)
        self._zoom_factor_base = 1.0015

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        assert event is not None
        # Update target positions if mouse moved significantly
        delta = self.target_viewport_pos - event.position()
        if abs(delta.x()) > 5 or abs(delta.y()) > 5:
            self.target_viewport_pos = event.position()
            self.target_scene_pos = self.mapToScene(event.position().toPoint())

        # Emit pixel coordinates
        scene_pos = self.mapToScene(event.position().toPoint())
        pixel_x = int(scene_pos.x())
        pixel_y = int(scene_pos.y())
        self.mousePositionChanged.emit(pixel_x, pixel_y)

        super().mouseMoveEvent(event)

    def get_zoom_level(self) -> float:
        # Get the current zoom level as a percentage (100.0 = 1:1)
        return self.transform().m11() * 100.0

    def gentle_zoom(self, factor: float) -> None:
        # Apply scale transformation
        self.scale(factor, factor)

        # Center on target scene position
        self.centerOn(self.target_scene_pos)

        # Calculate adjustment to keep mouse position stable
        viewport = self.viewport()
        assert viewport is not None
        delta_viewport_pos = self.target_viewport_pos - QtCore.QPointF(
            viewport.width() / 2.0, viewport.height() / 2.0
        )
        viewport_center = (
            QtCore.QPointF(self.mapFromScene(self.target_scene_pos)) - delta_viewport_pos
        )
        self.centerOn(self.mapToScene(viewport_center.toPoint()))

        # Emit zoom level change
        self.zoomChanged.emit(self.get_zoom_level())

    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        assert event is not None
        angle = event.angleDelta().y()
        factor = self._zoom_factor_base**angle
        self.gentle_zoom(factor)
        event.accept()

    def reset_zoom(self) -> None:
        # Reset transformation to identity
        self.setTransform(QtGui.QTransform())
        # Emit zoom level change
        self.zoomChanged.emit(100.0)

    def update_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        """Update the displayed pixmap while preserving zoom and viewport."""
        assert self.pixmap_item is not None
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())

    def enterEvent(self, event: QtGui.QEnterEvent | None) -> None:
        super().enterEvent(event)
        viewport = self.viewport()
        assert viewport is not None
        viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        super().mousePressEvent(event)
        viewport = self.viewport()
        assert viewport is not None
        viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        super().mouseReleaseEvent(event)
        viewport = self.viewport()
        assert viewport is not None
        viewport.setCursor(QtCore.Qt.CursorShape.ArrowCursor)


class InfoPanel(QtWidgets.QWidget):
    demosaicMethodChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        bgr888_buffer: np.ndarray,
        pixel_format,
        width: int,
        height: int,
        options: dict,
        filename: str | None,
    ) -> None:
        super().__init__()
        self.bgr888_buffer = bgr888_buffer
        self.pixel_format = pixel_format
        self.image_width = width
        self.image_height = height
        self.options = options
        self.filename = filename
        self.init_ui()

    def _get_available_demosaic_methods(self) -> list[str]:
        """Determine available demosaic methods based on format and backends."""
        from pixutils.formats.pixelformats import PixelColorEncoding
        from pixutils.conv.backends import get_backends

        # Only RAW formats support demosaic
        if self.pixel_format.color != PixelColorEncoding.RAW:
            return []

        # Get available backends (respecting options['backends'] if set)
        requested = self.options.get('backends')
        backends = get_backends(requested)

        methods = []

        # OpenCV only supports 'opencv' method for unpacked formats
        if 'opencv' in backends and not self.pixel_format.packed:
            methods.append('opencv')

        # Numba supports 3x3, bilinear, mosaic
        if 'numba' in backends:
            if '3x3' not in methods:
                methods.append('3x3')
            methods.append('bilinear')
            if 'mosaic' not in methods:
                methods.append('mosaic')

        # Numpy supports 3x3, mosaic
        if 'numpy' in backends:
            if '3x3' not in methods:
                methods.append('3x3')
            if 'mosaic' not in methods:
                methods.append('mosaic')

        return methods

    def _build_parameter_tree(self) -> list:
        """Build parameter tree structure for display."""
        param_list = []

        # Image Information group
        image_children = []
        if self.filename:
            basename = os.path.basename(self.filename)
            image_children.append(
                {'name': 'File', 'type': 'str', 'value': basename, 'readonly': True}
            )
        image_children.extend(
            [
                {
                    'name': 'Resolution',
                    'type': 'str',
                    'value': f'{self.image_width} × {self.image_height}',
                    'readonly': True,
                },
                {'name': 'Zoom', 'type': 'str', 'value': '100.0%', 'readonly': True},
                {'name': 'Pixel coords', 'type': 'str', 'value': '-', 'readonly': True},
                {'name': 'RGB', 'type': 'str', 'value': '-', 'readonly': True},
            ]
        )
        param_list.append(
            {'name': 'Image Information', 'type': 'group', 'children': image_children}
        )

        # Pixel Format group
        format_children = [
            {'name': 'Format', 'type': 'str', 'value': self.pixel_format.name, 'readonly': True},
            {
                'name': 'Color type',
                'type': 'str',
                'value': self.pixel_format.color.name,
                'readonly': True,
            },
        ]

        if self.pixel_format.drm_fourcc is not None:
            drm_str = fourcc_to_str(self.pixel_format.drm_fourcc)
            format_children.append(
                {'name': 'DRM FourCC', 'type': 'str', 'value': drm_str, 'readonly': True}
            )

        if self.pixel_format.v4l2_fourcc is not None:
            v4l2_str = fourcc_to_str(self.pixel_format.v4l2_fourcc)
            format_children.append(
                {'name': 'V4L2 FourCC', 'type': 'str', 'value': v4l2_str, 'readonly': True}
            )

        format_children.extend(
            [
                {
                    'name': 'Packed',
                    'type': 'str',
                    'value': 'Yes' if self.pixel_format.packed else 'No',
                    'readonly': True,
                },
                {
                    'name': 'Pixel alignment',
                    'type': 'str',
                    'value': f'{self.pixel_format.pixel_align[0]} × {self.pixel_format.pixel_align[1]}',
                    'readonly': True,
                },
            ]
        )

        if self.pixel_format.bayer_pattern is not None:
            format_children.append(
                {
                    'name': 'Bayer pattern',
                    'type': 'str',
                    'value': self.pixel_format.bayer_pattern,
                    'readonly': True,
                }
            )

        frame_size = self.pixel_format.framesize(self.image_width, self.image_height)
        format_children.extend(
            [
                {
                    'name': 'Planes',
                    'type': 'str',
                    'value': str(len(self.pixel_format.planes)),
                    'readonly': True,
                },
                {
                    'name': 'Frame size (bytes)',
                    'type': 'str',
                    'value': str(frame_size),
                    'readonly': True,
                },
            ]
        )

        param_list.append({'name': 'Pixel Format', 'type': 'group', 'children': format_children})

        # Plane groups
        for pi, plane in enumerate(self.pixel_format.planes):
            stride = self.pixel_format.stride(self.image_width, pi, 1)
            plane_size = self.pixel_format.planesize(stride, self.image_height, pi)

            plane_children = [
                {
                    'name': 'Bytes per block',
                    'type': 'str',
                    'value': str(plane.bytes_per_block),
                    'readonly': True,
                },
                {
                    'name': 'Pixels per block',
                    'type': 'str',
                    'value': str(plane.pixels_per_block),
                    'readonly': True,
                },
                {'name': 'Hsub', 'type': 'str', 'value': str(plane.hsub), 'readonly': True},
                {'name': 'Vsub', 'type': 'str', 'value': str(plane.vsub), 'readonly': True},
                {'name': 'Stride (bytes)', 'type': 'str', 'value': str(stride), 'readonly': True},
                {
                    'name': 'Plane size (bytes)',
                    'type': 'str',
                    'value': str(plane_size),
                    'readonly': True,
                },
            ]
            param_list.append({'name': f'Plane {pi}', 'type': 'group', 'children': plane_children})

        # Conversion Options group (conditional)
        methods = self._get_available_demosaic_methods()
        if self.options or methods:
            options_children = []

            if 'range' in self.options:
                options_children.append(
                    {
                        'name': 'Range',
                        'type': 'str',
                        'value': self.options['range'],
                        'readonly': True,
                    }
                )

            if 'encoding' in self.options:
                options_children.append(
                    {
                        'name': 'Encoding',
                        'type': 'str',
                        'value': self.options['encoding'],
                        'readonly': True,
                    }
                )

            if methods:
                # Build values list with "-" first
                demosaic_values = ['-'] + methods
                current = self.options.get('demosaic_method', '-')
                if current not in demosaic_values:
                    current = '-'
                options_children.append(
                    {
                        'name': 'Demosaic',
                        'type': 'list',
                        'values': demosaic_values,
                        'value': current,
                    }
                )

            if 'backends' in self.options:
                backends_str = ', '.join(self.options['backends'])
                options_children.append(
                    {'name': 'Backends', 'type': 'str', 'value': backends_str, 'readonly': True}
                )

            param_list.append(
                {'name': 'Conversion Options', 'type': 'group', 'children': options_children}
            )

        return param_list

    def _on_demosaic_param_changed(self, _, value: str) -> None:
        """Handle demosaic parameter change."""
        self.demosaicMethodChanged.emit(value)

    def init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()

        # Build parameter tree structure
        param_list = self._build_parameter_tree()

        # Create parameter tree
        self.params = Parameter.create(name='root', type='group', children=param_list)
        self.param_tree = ParameterTree(showHeader=False)
        self.param_tree.setParameters(self.params, showTop=False)

        layout.addWidget(self.param_tree)

        self.setLayout(layout)

        # Connect demosaic signal if available
        try:
            demosaic_param = self.params.child('Conversion Options', 'Demosaic')
            demosaic_param.sigValueChanged.connect(self._on_demosaic_param_changed)
        except Exception:
            pass  # No demosaic parameter for non-RAW formats

    def update_zoom_level(self, zoom_percent: float) -> None:
        zoom_param = self.params.child('Image Information', 'Zoom')
        zoom_param.setValue(f'{zoom_percent:.1f}%')

    def update_pixel_coords(self, x: int, y: int) -> None:
        coord_param = self.params.child('Image Information', 'Pixel coords')
        rgb_param = self.params.child('Image Information', 'RGB')
        if 0 <= x < self.image_width and 0 <= y < self.image_height:
            coord_param.setValue(f'{x}, {y}')
            # Extract RGB from BGR888 buffer
            bgr = self.bgr888_buffer[y, x]
            rgb_param.setValue(f'{bgr[0]}, {bgr[1]}, {bgr[2]}')
        else:
            coord_param.setValue('-')
            rgb_param.setValue('-')

    def update_bgr888_buffer(self, new_buffer: np.ndarray) -> None:
        """Update the BGR888 buffer used for pixel color lookups."""
        self.bgr888_buffer = new_buffer


class ImageViewerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        buffer: np.ndarray,
        pixel_format,
        width: int,
        height: int,
        bytesperline: int,
        options: dict,
        filename: str | None,
        title: str = 'Image Viewer',
    ) -> None:
        super().__init__()
        self.buffer = buffer
        self.pixel_format = pixel_format
        self.image_width = width
        self.image_height = height
        self.bytesperline = bytesperline
        self.options = options.copy()
        self.filename = filename
        self.bgr888_buffer = None

        pixmap = self._perform_conversion()
        assert self.bgr888_buffer is not None
        self.image_widget = ZoomableGraphicsView(pixmap)
        self.info_panel = InfoPanel(
            self.bgr888_buffer, pixel_format, width, height, options, filename
        )

        # Connect signals to info panel
        self.image_widget.zoomChanged.connect(self.info_panel.update_zoom_level)
        self.image_widget.mousePositionChanged.connect(self.info_panel.update_pixel_coords)

        # Connect demosaic method change signal
        self.info_panel.demosaicMethodChanged.connect(self._on_demosaic_changed)

        self.init_ui()
        self.setWindowTitle(title)

    def _perform_conversion(self) -> QtGui.QPixmap:
        """Convert raw buffer to BGR888 and create pixmap."""
        from pixutils.conv.conv import buffer_to_bgr888
        from pixutils.conv.qt import bgr888_to_pix

        self.bgr888_buffer = buffer_to_bgr888(
            self.pixel_format,
            self.image_width,
            self.image_height,
            self.bytesperline,
            self.buffer,
            self.options,
        )
        return bgr888_to_pix(self.bgr888_buffer)

    def _on_demosaic_changed(self, new_method: str) -> None:
        """Handle demosaic method change by re-converting the image."""
        from pixutils.conv.conv import buffer_to_bgr888
        from pixutils.conv.qt import bgr888_to_pix

        # Store current method for error recovery
        old_method = self.options.get('demosaic_method')

        # Update options: "-" means use default (remove from options)
        if new_method == '-':
            self.options.pop('demosaic_method', None)
        else:
            self.options['demosaic_method'] = new_method

        # Show busy cursor and disable dropdown
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            demosaic_param = self.info_panel.params.child('Conversion Options', 'Demosaic')
            demosaic_param.setOpts(enabled=False)
        except Exception:
            pass

        try:
            # Re-convert with new method
            new_bgr888 = buffer_to_bgr888(
                self.pixel_format,
                self.image_width,
                self.image_height,
                self.bytesperline,
                self.buffer,
                self.options,
            )

            # Convert to pixmap
            new_pixmap = bgr888_to_pix(new_bgr888)

            # Update display
            self.image_widget.update_pixmap(new_pixmap)

            # Update info panel buffer for pixel color lookups
            self.info_panel.update_bgr888_buffer(new_bgr888)

            # Store new buffer
            self.bgr888_buffer = new_bgr888

        except Exception as e:
            # Show error dialog
            QtWidgets.QMessageBox.critical(
                self,
                'Conversion Error',
                f'Failed to convert image with demosaic method "{new_method}":\n\n{str(e)}',
            )

            # Revert to old method
            if old_method:
                self.options['demosaic_method'] = old_method
            else:
                self.options.pop('demosaic_method', None)
            try:
                demosaic_param = self.info_panel.params.child('Conversion Options', 'Demosaic')
                demosaic_param.setValue(old_method if old_method else '-')
            except Exception:
                pass

        finally:
            # Restore cursor and re-enable dropdown
            QtWidgets.QApplication.restoreOverrideCursor()
            try:
                demosaic_param = self.info_panel.params.child('Conversion Options', 'Demosaic')
                demosaic_param.setOpts(enabled=True)
            except Exception:
                pass

    def init_ui(self) -> None:
        # Create horizontal splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.info_panel)
        splitter.addWidget(self.image_widget)

        # Set initial sizes: info panel 250px, image widget 550px
        splitter.setSizes([250, 550])

        self.setCentralWidget(splitter)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        assert event is not None
        if event.key() == QtCore.Qt.Key.Key_R:
            self.image_widget.reset_zoom()
        else:
            super().keyPressEvent(event)


def parse_filename_heuristics(filename):
    """Try to parse width, height, format, range and encoding from filename"""
    basename = os.path.basename(filename)
    if basename.endswith('.gz'):
        basename = os.path.splitext(basename)[0]

    basename_no_ext = os.path.splitext(basename)[0]

    parts = re.split(r'[-_]', basename_no_ext)
    parts = [p for p in parts if p]

    if not parts:
        return None

    encoding_options = {'bt601', 'bt709', 'bt2020'}
    range_options = {'full', 'limited'}

    result = {}

    remaining_parts = parts[:]

    for part in parts:
        part_lower = part.lower()
        if part_lower in encoding_options:
            result['encoding'] = part_lower
            remaining_parts.remove(part)
        elif part_lower in range_options:
            result['range'] = part_lower
            remaining_parts.remove(part)

    for part in remaining_parts[:]:
        try:
            result['format'] = PixelFormats.find_by_name(part)
            remaining_parts.remove(part)
            break
        except StopIteration:
            continue

    if 'format' not in result:
        return None

    for part in remaining_parts:
        if 'x' in part:
            try:
                w, h = part.split('x', 1)
                result['width'] = int(w)
                result['height'] = int(h)
                break
            except (ValueError, IndexError):
                continue

    if 'width' not in result or 'height' not in result:
        return None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument(
        'width', nargs='?', help='Width in pixels (optional if filename contains heuristics)'
    )
    parser.add_argument(
        'height', nargs='?', help='Height in pixels (optional if filename contains heuristics)'
    )
    parser.add_argument(
        'format', nargs='?', help='Pixel format (optional if filename contains heuristics)'
    )
    parser.add_argument('--range', choices=['full', 'limited'], help='Color range')
    parser.add_argument('--encoding', choices=['bt601', 'bt709', 'bt2020'], help='Color encoding')
    parser.add_argument('--demosaic', choices=['3x3', 'bilinear', 'mosaic'], help='Demosaic method')
    parser.add_argument(
        '--backends',
        type=str,
        default=None,
        help='Comma-separated list of backends in priority order',
    )
    args = parser.parse_args()

    if args.width and args.height and args.format:
        format = PixelFormats.find_by_name(args.format)
        w = int(args.width)
        h = int(args.height)
        detected_encoding = args.encoding
        detected_range = args.range
    else:
        parsed = parse_filename_heuristics(args.file)
        if parsed is None:
            if not args.width or not args.height or not args.format:
                parser.error(
                    'Could not detect parameters from filename. Please provide width, height, and format arguments.'
                )
            format = PixelFormats.find_by_name(args.format)
            w = int(args.width)
            h = int(args.height)
            detected_encoding = args.encoding
            detected_range = args.range
        else:
            format = parsed['format']
            w = parsed['width']
            h = parsed['height']
            detected_encoding = parsed.get('encoding')
            detected_range = parsed.get('range')

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

    options = {}
    final_range = args.range if args.range else detected_range
    final_encoding = args.encoding if args.encoding else detected_encoding

    if final_range:
        options['range'] = final_range
    if final_encoding:
        options['encoding'] = final_encoding
    if args.demosaic:
        options['demosaic_method'] = args.demosaic
    if args.backends:
        options['backends'] = [b.strip() for b in args.backends.split(',')]

    # Handle stdin case: args.file can be '-' or None
    filename = args.file if args.file and args.file != '-' else None

    window = ImageViewerWindow(
        buffer=buf,
        pixel_format=format,
        width=w,
        height=h,
        bytesperline=0,
        options=options,
        filename=filename,
        title=f'View {filename}',
    )
    window.resize(1600, 1000)
    window.show()

    qapp.exec()


if __name__ == '__main__':
    main()
