# XXX I have not been able to get PyQt6 imported to pylint
# pylint: skip-file

# Qt methods give false positives about incompatible overrides
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

from pixutils.formats import fourcc_to_str, PixelFormat


class ZoomableGraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = QtCore.pyqtSignal(float)
    mousePositionChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, pixmap: QtGui.QPixmap | None = None) -> None:
        super().__init__()

        # Create scene and optionally add pixmap
        self._scene = QtWidgets.QGraphicsScene()
        self._setup_checkerboard_background()
        self.pixmap_item = self._scene.addPixmap(pixmap) if pixmap else None
        self.setScene(self._scene)

        # Configure view
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)

        # Disable scroll bars - we use drag to pan
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Track mouse position for zoom-under-cursor
        self.target_viewport_pos = QtCore.QPointF()
        self.target_scene_pos = QtCore.QPointF()
        self.setMouseTracking(True)

        # Base for exponential zoom calculation: raised to wheel angle delta to get smooth,
        # proportional zoom (larger wheel movements = larger zoom changes)
        self._zoom_factor_base = 1.0015

    def _setup_checkerboard_background(self) -> None:
        """Create a checkerboard pattern for the scene background."""
        # Create a small pixmap with a 2x2 checkerboard pattern
        checker_size = 16
        checker_pixmap = QtGui.QPixmap(checker_size * 2, checker_size * 2)
        painter = QtGui.QPainter(checker_pixmap)

        # Light gray and slightly darker gray for the checkerboard
        color1 = QtGui.QColor(128, 128, 128)
        color2 = QtGui.QColor(100, 100, 100)

        painter.fillRect(0, 0, checker_size, checker_size, color1)
        painter.fillRect(checker_size, 0, checker_size, checker_size, color2)
        painter.fillRect(0, checker_size, checker_size, checker_size, color2)
        painter.fillRect(checker_size, checker_size, checker_size, checker_size, color1)
        painter.end()

        # Set the tiled brush as the scene background
        self._scene.setBackgroundBrush(QtGui.QBrush(checker_pixmap))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        assert event is not None
        # Update target positions if mouse moved significantly
        delta = self.target_viewport_pos - event.position()
        if abs(delta.x()) > 5 or abs(delta.y()) > 5:
            self.target_viewport_pos = event.position()
            self.target_scene_pos = self.mapToScene(event.position().toPoint())

        # Emit pixel coordinates
        scene_pos = self.mapToScene(event.position().toPoint())
        pixel_x, pixel_y = self.scene_to_pixel_coords(scene_pos)
        self.mousePositionChanged.emit(pixel_x, pixel_y)

        super().mouseMoveEvent(event)

    def get_zoom_level(self) -> float:
        # Get the current zoom level as a percentage (100.0 = 1:1)
        return self.transform().m11() * 100.0

    def scene_to_pixel_coords(self, scene_pos: QtCore.QPointF) -> tuple[int, int]:
        """Convert scene coordinates to pixel coordinates in the image."""
        if self.pixmap_item is None:
            return (int(scene_pos.x()), int(scene_pos.y()))
        # Pixmap is centered at origin, so pixel (0,0) is at scene (-width/2, -height/2)
        rect = self.pixmap_item.boundingRect()
        pixel_x = int(scene_pos.x() + rect.width() / 2)
        pixel_y = int(scene_pos.y() + rect.height() / 2)
        return (pixel_x, pixel_y)

    def _update_scene_rect(self) -> None:
        """Expand scene rect to allow free panning even when image is smaller than viewport."""
        if self.pixmap_item is None:
            return
        # Get image bounds in scene coordinates (accounts for centering offset)
        pixmap_rect = self.pixmap_item.sceneBoundingRect()
        # Get viewport extent in scene coordinates
        viewport = self.viewport()
        assert viewport is not None
        viewport_rect = self.mapToScene(viewport.rect()).boundingRect()
        # Expand by one viewport width/height on each side
        expanded = pixmap_rect.adjusted(
            -viewport_rect.width(),
            -viewport_rect.height(),
            viewport_rect.width(),
            viewport_rect.height(),
        )
        self.setSceneRect(expanded)

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
        self._update_scene_rect()

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
        self._update_scene_rect()

    def fit_to_view(self, padding: float = 0.98) -> None:
        """Fit the entire pixmap in viewport while preserving aspect ratio.

        Args:
            padding: Scale factor for margin (0.98 = 2% padding). Range: 0 < padding <= 1.0
        """
        # Validate preconditions
        if self.pixmap_item is None:
            return

        # Validate padding parameter
        if padding <= 0 or padding > 1.0:
            return

        # Get viewport dimensions
        viewport = self.viewport()
        assert viewport is not None
        viewport_width = viewport.width()
        viewport_height = viewport.height()

        # Get pixmap dimensions
        pixmap_rect = self.pixmap_item.sceneBoundingRect()
        pixmap_width = pixmap_rect.width()
        pixmap_height = pixmap_rect.height()

        # Handle edge cases with zero/negative dimensions
        if pixmap_width <= 0 or pixmap_height <= 0:
            return

        # Calculate scale to fit while preserving aspect ratio
        scale_x = viewport_width / pixmap_width
        scale_y = viewport_height / pixmap_height
        target_scale = min(scale_x, scale_y) * padding

        # Apply transform
        self.setTransform(QtGui.QTransform())
        self.scale(target_scale, target_scale)

        # Center view (pixmap is centered at scene origin)
        self.centerOn(0, 0)

        # Emit zoom level change
        self.zoomChanged.emit(target_scale * 100.0)

        # Update scene rect
        self._update_scene_rect()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_scene_rect()

    def update_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        """Update the displayed pixmap while preserving zoom and viewport."""
        if self.pixmap_item is None:
            self.pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self.pixmap_item.setPixmap(pixmap)
        assert self.pixmap_item is not None
        # Center the pixmap at scene origin
        rect = self.pixmap_item.boundingRect()
        self.pixmap_item.setOffset(-rect.width() / 2, -rect.height() / 2)
        self._update_scene_rect()

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
    collapseChanged = QtCore.pyqtSignal(bool)  # True = collapsed, False = expanded
    COLLAPSE_BUTTON_SIZE = 30

    def __init__(self, options: dict) -> None:
        super().__init__()
        self.options = options
        self.pixel_format: PixelFormat | None = None
        self.image_width = 0
        self.image_height = 0
        self.bgr888_buffer: np.ndarray | None = None
        self._is_collapsed = False
        self._create_ui()

    def _get_all_demosaic_methods(self) -> list[str]:
        """Get all demosaic methods available from installed backends."""
        from pixutils.conv.backends import get_backends

        requested = self.options.get('backends')
        backends = get_backends(requested)

        methods = []

        if 'opencv' in backends:
            methods.append('opencv')

        if 'numba' in backends:
            if '3x3' not in methods:
                methods.append('3x3')
            methods.append('bilinear')
            if 'mosaic' not in methods:
                methods.append('mosaic')

        if 'numpy' in backends:
            if '3x3' not in methods:
                methods.append('3x3')
            if 'mosaic' not in methods:
                methods.append('mosaic')

        return methods

    def _build_static_param_tree(self) -> list:
        """Build the complete static parameter tree structure."""
        plane_fields = [
            {'name': 'Bytes per block', 'type': 'str', 'value': '-', 'readonly': True},
            {'name': 'Pixels per block', 'type': 'str', 'value': '-', 'readonly': True},
            {'name': 'Hsub', 'type': 'str', 'value': '-', 'readonly': True},
            {'name': 'Vsub', 'type': 'str', 'value': '-', 'readonly': True},
            {'name': 'Stride (bytes)', 'type': 'str', 'value': '-', 'readonly': True},
            {'name': 'Plane size (bytes)', 'type': 'str', 'value': '-', 'readonly': True},
        ]

        return [
            {
                'name': 'Image Information',
                'type': 'group',
                'children': [
                    {'name': 'Resolution', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Zoom', 'type': 'str', 'value': '100.0%', 'readonly': True},
                    {'name': 'Pixel coords', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'RGB', 'type': 'str', 'value': '-', 'readonly': True},
                ],
            },
            {
                'name': 'Pixel Format',
                'type': 'group',
                'children': [
                    {'name': 'Format', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Color type', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'DRM FourCC', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'V4L2 FourCC', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Packed', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Pixel alignment', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Bayer pattern', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Planes', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Frame size (bytes)', 'type': 'str', 'value': '-', 'readonly': True},
                ],
            },
            {
                'name': 'Plane 0',
                'type': 'group',
                'children': [dict(f) for f in plane_fields],
            },
            {
                'name': 'Plane 1',
                'type': 'group',
                'children': [dict(f) for f in plane_fields],
            },
            {
                'name': 'Plane 2',
                'type': 'group',
                'children': [dict(f) for f in plane_fields],
            },
            {
                'name': 'Conversion Options',
                'type': 'group',
                'children': [
                    {'name': 'Range', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Encoding', 'type': 'str', 'value': '-', 'readonly': True},
                    {'name': 'Demosaic', 'type': 'list', 'values': ['-'], 'value': '-'},
                    {'name': 'Backends', 'type': 'str', 'value': '-', 'readonly': True},
                ],
            },
        ]

    def _on_demosaic_param_changed(self, _, value: str) -> None:
        """Handle demosaic parameter change."""
        self.demosaicMethodChanged.emit(value)

    def _on_collapse_button_clicked(self) -> None:
        """Toggle collapsed state."""
        self._is_collapsed = not self._is_collapsed
        self.param_tree.setVisible(not self._is_collapsed)
        self._collapse_button.setText('▶' if self._is_collapsed else '◀')
        self.collapseChanged.emit(self._is_collapsed)

    def _create_ui(self) -> None:
        """Create layout, build static parameter tree, and connect signals."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Add collapse button at the top, aligned to the right
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()
        self._collapse_button = QtWidgets.QPushButton('◀')
        self._collapse_button.setFixedSize(self.COLLAPSE_BUTTON_SIZE, self.COLLAPSE_BUTTON_SIZE)
        self._collapse_button.clicked.connect(self._on_collapse_button_clicked)
        button_layout.addWidget(self._collapse_button)
        layout.addLayout(button_layout)

        # Add parameter tree
        self.param_tree = ParameterTree(showHeader=False)
        # self.param_tree.setMinimumWidth(0)
        layout.addWidget(self.param_tree)

        # Add stretch to keep button at top when param_tree is hidden
        layout.addStretch()

        self.setLayout(layout)

        # Allow the panel to shrink down to just the button width
        self.setMinimumWidth(0)

        param_list = self._build_static_param_tree()
        self.params = Parameter.create(name='root', type='group', children=param_list)
        self.param_tree.setParameters(self.params, showTop=False)

        # Populate demosaic dropdown once with all methods from installed
        # backends. The selection is ignored for non-RAW formats at conversion
        # time, but the list never changes.
        demosaic_param = self.params.child('Conversion Options', 'Demosaic')
        methods = self._get_all_demosaic_methods()
        if methods:
            demosaic_values = ['-'] + methods
            current = self.options.get('demosaic_method', '-')
            if current not in demosaic_values:
                current = '-'
            demosaic_param.setLimits(demosaic_values)
            demosaic_param.setValue(current)
        demosaic_param.sigValueChanged.connect(self._on_demosaic_param_changed)

    def _update_format_values(self) -> None:
        """Update parameter values to reflect current pixel_format and options."""
        assert self.pixel_format is not None

        # Image Information
        res_param = self.params.child('Image Information', 'Resolution')
        res_param.setValue(f'{self.image_width} × {self.image_height}')

        # Pixel Format
        pf = self.pixel_format
        fmt_group = self.params.child('Pixel Format')
        fmt_group.child('Format').setValue(pf.name)
        fmt_group.child('Color type').setValue(pf.color.name)
        fmt_group.child('DRM FourCC').setValue(
            fourcc_to_str(pf.drm_fourcc) if pf.drm_fourcc is not None else '-'
        )
        fmt_group.child('V4L2 FourCC').setValue(
            fourcc_to_str(pf.v4l2_fourcc) if pf.v4l2_fourcc is not None else '-'
        )
        fmt_group.child('Packed').setValue('Yes' if pf.csi2_packed else 'No')
        fmt_group.child('Pixel alignment').setValue(f'{pf.pixel_align[0]} × {pf.pixel_align[1]}')
        fmt_group.child('Bayer pattern').setValue(
            pf.bayer_pattern if pf.bayer_pattern is not None else '-'
        )
        fmt_group.child('Planes').setValue(str(len(pf.planes)))
        fmt_group.child('Frame size (bytes)').setValue(
            str(pf.framesize(self.image_width, self.image_height))
        )

        # Plane groups — fill active planes, blank inactive ones
        for pi in range(3):
            plane_group = self.params.child(f'Plane {pi}')
            if pi < len(pf.planes):
                plane = pf.planes[pi]
                stride = pf.stride(self.image_width, pi, 1)
                plane_size = pf.planesize(stride, self.image_height, pi)
                plane_group.child('Bytes per block').setValue(str(plane.bytes_per_block))
                plane_group.child('Pixels per block').setValue(str(plane.pixels_per_block))
                plane_group.child('Hsub').setValue(str(plane.hsub))
                plane_group.child('Vsub').setValue(str(plane.vsub))
                plane_group.child('Stride (bytes)').setValue(str(stride))
                plane_group.child('Plane size (bytes)').setValue(str(plane_size))
            else:
                for child in plane_group.children():
                    child.setValue('-')

        # Conversion Options
        opts_group = self.params.child('Conversion Options')
        opts_group.child('Range').setValue(self.options.get('range', '-'))
        opts_group.child('Encoding').setValue(self.options.get('encoding', '-'))
        opts_group.child('Backends').setValue(
            ', '.join(self.options['backends']) if 'backends' in self.options else '-'
        )

    def configure(self, pixel_format, width: int, height: int) -> None:
        """Update format/size and refresh parameter values."""
        self.pixel_format = pixel_format
        self.image_width = width
        self.image_height = height
        self._update_format_values()

    def update_zoom_level(self, zoom_percent: float) -> None:
        zoom_param = self.params.child('Image Information', 'Zoom')
        zoom_param.setValue(f'{zoom_percent:.1f}%')

    def update_pixel_coords(self, x: int, y: int) -> None:
        coord_param = self.params.child('Image Information', 'Pixel coords')
        rgb_param = self.params.child('Image Information', 'RGB')
        if 0 <= x < self.image_width and 0 <= y < self.image_height:
            coord_param.setValue(f'{x}, {y}')
            # Extract RGB from BGR888 buffer
            assert self.bgr888_buffer is not None
            bgr = self.bgr888_buffer[y, x]
            rgb_param.setValue(f'{bgr[0]}, {bgr[1]}, {bgr[2]}')
        else:
            coord_param.setValue('-')
            rgb_param.setValue('-')

    def update_bgr888_buffer(self, new_buffer: np.ndarray) -> None:
        """Update the BGR888 buffer used for pixel color lookups."""
        self.bgr888_buffer = new_buffer


class TitleBar(QtWidgets.QWidget):
    BAR_HEIGHT = 30

    def __init__(self) -> None:
        super().__init__()
        self._resolution_text = '-'
        self._format_text = '-'
        self._frames_text = '0'
        self._fps_text = '-'
        self._create_ui()

    def _create_ui(self) -> None:
        """Create horizontal layout with resolution and format labels."""
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Left stretch to center content
        layout.addStretch()

        # Resolution label
        self._resolution_label = QtWidgets.QLabel(self._resolution_text)
        layout.addWidget(self._resolution_label)

        # Separator
        separator1 = QtWidgets.QLabel(' | ')
        layout.addWidget(separator1)

        # Format label
        self._format_label = QtWidgets.QLabel(self._format_text)
        layout.addWidget(self._format_label)

        # Separator
        separator2 = QtWidgets.QLabel(' | ')
        layout.addWidget(separator2)

        # Frames label
        self._frames_label = QtWidgets.QLabel(self._frames_text)
        layout.addWidget(self._frames_label)

        # Separator
        separator3 = QtWidgets.QLabel(' | ')
        layout.addWidget(separator3)

        # FPS label
        self._fps_label = QtWidgets.QLabel(self._fps_text)
        layout.addWidget(self._fps_label)

        # Right stretch to center content
        layout.addStretch()

        self.setLayout(layout)
        self.setFixedHeight(self.BAR_HEIGHT)

    def update_info(self, width: int, height: int, format_name: str) -> None:
        """Update displayed resolution and format information."""
        self._resolution_text = f'{width} × {height}'
        self._format_text = format_name
        self._resolution_label.setText(self._resolution_text)
        self._format_label.setText(self._format_text)

    def update_frame_stats(self, frame_count: int, fps_text: str) -> None:
        """Update displayed frame count and FPS."""
        self._frames_text = str(frame_count)
        self._fps_text = fps_text
        self._frames_label.setText(self._frames_text)
        self._fps_label.setText(self._fps_text)


@dataclass
class StreamState:
    # Widgets
    view: ZoomableGraphicsView
    title_bar: TitleBar
    container: QtWidgets.QWidget

    # Image state
    pixel_format: PixelFormat | None = None
    width: int = 0
    height: int = 0
    bytesperline: int = 0
    buffer: np.ndarray | None = None
    bgr888_buffer: np.ndarray | None = None

    # Frame statistics
    frame_count: int = 0
    last_frame_time: float | None = None


class ImageViewerWidget(QtWidgets.QWidget):
    def __init__(self, options: dict) -> None:
        super().__init__()
        self.options = options.copy()
        self._panel_expanded_width = 250

        # Multi-stream state
        self.streams: list[StreamState] = []
        self.active_stream_index: int = 0
        self.num_streams: int = 1

        self.info_panel = InfoPanel(options)

        # Connect demosaic method change signal
        self.info_panel.demosaicMethodChanged.connect(self._on_demosaic_changed)

        # Connect collapse signal
        self.info_panel.collapseChanged.connect(self._on_panel_collapsed)

        self._init_ui()

    def _create_stream(self) -> StreamState:
        """Create a new stream with view, title bar, and container."""
        view = ZoomableGraphicsView()
        title_bar = TitleBar()

        # Create container for title bar and view
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(title_bar)
        container_layout.addWidget(view)

        # Set size policy to allow container to expand and fill available space
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

        # Set border style (will be updated when active)
        container.setStyleSheet('border: 2px solid transparent;')

        # Install event filter for hover/click detection
        viewport = view.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)

        return StreamState(view=view, title_bar=title_bar, container=container)

    def _destroy_stream(self, stream: StreamState) -> None:
        """Cleanup stream widgets and event filters."""
        viewport = stream.view.viewport()
        if viewport is not None:
            viewport.removeEventFilter(self)
        stream.container.deleteLater()

    def setNumStreams(self, count: int) -> None:
        """Set the number of streams to display."""
        if count < 1 or count > 4:
            raise ValueError('Stream count must be between 1 and 4')

        # Disconnect signals from current active stream
        if self.streams:
            self._disconnect_stream_signals(self.active_stream_index)

        # Add new streams
        while len(self.streams) < count:
            stream = self._create_stream()
            self.streams.append(stream)

        # Remove excess streams
        while len(self.streams) > count:
            stream = self.streams.pop()
            self._destroy_stream(stream)

        self.num_streams = count

        # Reset active stream if it's now out of range
        if self.active_stream_index >= count:
            self.active_stream_index = 0

        # Rebuild grid layout
        self._rebuild_grid_layout()

        # Connect signals for new active stream
        if self.streams:
            self._connect_stream_signals(self.active_stream_index)
            self._set_active_stream(self.active_stream_index)

    def _rebuild_grid_layout(self) -> None:
        """Arrange streams in grid based on count."""
        # Clear existing layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        # Grid patterns: 1=1x1, 2=2x1, 3=2x2 (bottom-right empty), 4=2x2
        if self.num_streams == 1:
            self.grid_layout.addWidget(self.streams[0].container, 0, 0)
        elif self.num_streams == 2:
            self.grid_layout.addWidget(self.streams[0].container, 0, 0)
            self.grid_layout.addWidget(self.streams[1].container, 0, 1)
        elif self.num_streams == 3:
            self.grid_layout.addWidget(self.streams[0].container, 0, 0)
            self.grid_layout.addWidget(self.streams[1].container, 0, 1)
            self.grid_layout.addWidget(self.streams[2].container, 1, 0)
        elif self.num_streams == 4:
            self.grid_layout.addWidget(self.streams[0].container, 0, 0)
            self.grid_layout.addWidget(self.streams[1].container, 0, 1)
            self.grid_layout.addWidget(self.streams[2].container, 1, 0)
            self.grid_layout.addWidget(self.streams[3].container, 1, 1)

    def _connect_stream_signals(self, stream_index: int) -> None:
        """Connect stream's view signals to InfoPanel."""
        if stream_index < 0 or stream_index >= len(self.streams):
            return
        stream = self.streams[stream_index]
        stream.view.zoomChanged.connect(self.info_panel.update_zoom_level)
        stream.view.mousePositionChanged.connect(self._on_stream_mouse_moved)

    def _disconnect_stream_signals(self, stream_index: int) -> None:
        """Disconnect stream's view signals from InfoPanel."""
        if stream_index < 0 or stream_index >= len(self.streams):
            return
        stream = self.streams[stream_index]
        try:
            stream.view.zoomChanged.disconnect(self.info_panel.update_zoom_level)
            stream.view.mousePositionChanged.disconnect(self._on_stream_mouse_moved)
        except TypeError:
            pass

    def _on_stream_mouse_moved(self, x: int, y: int) -> None:
        """Route mouse position to InfoPanel with pixel color lookup."""
        self.info_panel.update_pixel_coords(x, y)

    def _set_active_stream(self, stream_index: int) -> None:
        """Set the active stream, update borders, and refresh InfoPanel."""
        if stream_index < 0 or stream_index >= len(self.streams):
            return

        # Update borders
        for i, stream in enumerate(self.streams):
            if i == stream_index:
                stream.container.setStyleSheet('border: 2px solid #4A9EFF;')
            else:
                stream.container.setStyleSheet('border: 2px solid transparent;')

        self.active_stream_index = stream_index
        self._update_infopanel_for_stream(stream_index)

    def _update_infopanel_for_stream(self, stream_index: int) -> None:
        """Configure InfoPanel to display data for the specified stream."""
        if stream_index < 0 or stream_index >= len(self.streams):
            return

        stream = self.streams[stream_index]

        # Update format info if stream has data
        if stream.pixel_format is not None and stream.bgr888_buffer is not None:
            self.info_panel.configure(stream.pixel_format, stream.width, stream.height)
            self.info_panel.update_bgr888_buffer(stream.bgr888_buffer)
            self.info_panel.update_zoom_level(stream.view.get_zoom_level())

            # Update pixel coords at current mouse position
            pos = stream.view.target_scene_pos
            pixel_x, pixel_y = stream.view.scene_to_pixel_coords(pos)
            self.info_panel.update_pixel_coords(pixel_x, pixel_y)

    def eventFilter(self, obj: QtCore.QObject | None, event: QtCore.QEvent | None) -> bool:
        """Handle mouse enter/leave/click events on stream viewports for hover/focus."""
        if event is None:
            return super().eventFilter(obj, event)

        # Find which stream this viewport belongs to
        stream_index = -1
        for i, stream in enumerate(self.streams):
            if obj == stream.view.viewport():
                stream_index = i
                break

        if stream_index == -1:
            return super().eventFilter(obj, event)

        event_type = event.type()

        # Mouse enter: temporarily show this stream's data
        if event_type == QtCore.QEvent.Type.Enter:
            if stream_index != self.active_stream_index:
                # Disconnect active stream signals
                self._disconnect_stream_signals(self.active_stream_index)
                # Connect hovered stream signals
                self._connect_stream_signals(stream_index)
                # Update InfoPanel
                self._update_infopanel_for_stream(stream_index)

        # Mouse leave: restore active stream's data
        elif event_type == QtCore.QEvent.Type.Leave:
            if stream_index != self.active_stream_index:
                # Disconnect hovered stream signals
                self._disconnect_stream_signals(stream_index)
                # Reconnect active stream signals
                self._connect_stream_signals(self.active_stream_index)
                # Restore InfoPanel
                self._update_infopanel_for_stream(self.active_stream_index)

        # Mouse click: make this stream active
        elif event_type == QtCore.QEvent.Type.MouseButtonPress:
            if stream_index != self.active_stream_index:
                # Disconnect old active stream
                self._disconnect_stream_signals(self.active_stream_index)
                # Set new active stream
                self._set_active_stream(stream_index)
                # Connect new active stream
                self._connect_stream_signals(stream_index)

        return super().eventFilter(obj, event)

    def _convert_buffer_for_stream(self, stream: StreamState) -> tuple[np.ndarray, QtGui.QPixmap]:
        """Convert a stream's raw buffer to BGR888 array and QPixmap."""
        from pixutils.conv.conv import buffer_to_bgr888
        from pixutils.conv.qt import bgr888_to_pix

        assert stream.pixel_format is not None
        assert stream.buffer is not None
        bgr888 = buffer_to_bgr888(
            stream.pixel_format,
            stream.width,
            stream.height,
            stream.bytesperline,
            stream.buffer,
            self.options,
        )
        return bgr888, bgr888_to_pix(bgr888)

    def _calculate_frame_stats_for_stream(self, stream: StreamState) -> tuple[int, str]:
        """Calculate frame count and FPS for a stream. Returns (frame_count, fps_text)."""
        now = time.monotonic()
        stream.frame_count += 1

        if stream.last_frame_time is not None:
            dt = now - stream.last_frame_time
            fps_text = f'{1.0 / dt:.1f}' if dt > 0 else '-'
        else:
            fps_text = '-'

        stream.last_frame_time = now
        return stream.frame_count, fps_text

    def set_frame(
        self,
        stream_num: int,
        width: int,
        height: int,
        pixel_format,
        buffer: np.ndarray,
        bytesperline: int = 0,
    ) -> None:
        """Provide a new frame for the specified stream. Updates InfoPanel if stream is active."""
        if stream_num < 0 or stream_num >= len(self.streams):
            raise ValueError(f'Invalid stream_num {stream_num}, must be 0-{len(self.streams) - 1}')

        stream = self.streams[stream_num]

        format_changed = (
            pixel_format != stream.pixel_format or width != stream.width or height != stream.height
        )

        # Update stream state
        stream.pixel_format = pixel_format
        stream.width = width
        stream.height = height
        stream.bytesperline = bytesperline
        stream.buffer = buffer

        # Convert and display
        stream.bgr888_buffer, pixmap = self._convert_buffer_for_stream(stream)
        stream.view.update_pixmap(pixmap)

        # Apply fit on first frame or resolution change
        if format_changed:
            stream.view.fit_to_view()

        # Update stream's title bar
        if format_changed:
            stream.title_bar.update_info(width, height, pixel_format.name)

        # Calculate frame stats and update title bar
        frame_count, fps_text = self._calculate_frame_stats_for_stream(stream)
        stream.title_bar.update_frame_stats(frame_count, fps_text)

        # If this is the active stream, update InfoPanel
        if stream_num == self.active_stream_index:
            if format_changed:
                self.info_panel.configure(pixel_format, width, height)

            self.info_panel.update_bgr888_buffer(stream.bgr888_buffer)

            # Re-sample RGB at the last known mouse position
            pos = stream.view.target_scene_pos
            pixel_x, pixel_y = stream.view.scene_to_pixel_coords(pos)
            self.info_panel.update_pixel_coords(pixel_x, pixel_y)

    def _on_demosaic_changed(self, new_method: str) -> None:
        """Handle demosaic method change by re-converting all streams."""
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
            # Re-convert all streams that have data
            for stream in self.streams:
                if stream.buffer is not None:
                    new_bgr888, new_pixmap = self._convert_buffer_for_stream(stream)
                    stream.view.update_pixmap(new_pixmap)
                    stream.bgr888_buffer = new_bgr888

            # Update active stream's InfoPanel buffer
            active_stream = self.streams[self.active_stream_index]
            if active_stream.bgr888_buffer is not None:
                self.info_panel.update_bgr888_buffer(active_stream.bgr888_buffer)

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

    def _on_panel_collapsed(self, is_collapsed: bool) -> None:
        """Handle panel collapse/expand from the InfoPanel."""
        current_sizes = self.splitter.sizes()
        collapsed_width = InfoPanel.COLLAPSE_BUTTON_SIZE
        if is_collapsed:
            # Collapse: save current width, resize to minimal size (button width)
            self._panel_expanded_width = current_sizes[0]
            self.splitter.setSizes(
                [collapsed_width, current_sizes[0] + current_sizes[1] - collapsed_width]
            )
        else:
            # Expand: restore saved width, subtract from image widget
            self.splitter.setSizes(
                [
                    self._panel_expanded_width,
                    current_sizes[1] - self._panel_expanded_width + collapsed_width,
                ]
            )

    def _init_ui(self) -> None:
        # Create initial stream
        self.streams.append(self._create_stream())
        self._connect_stream_signals(0)

        # Create grid container for streams
        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.grid_layout.addWidget(self.streams[0].container, 0, 0)

        # Create splitter with info panel and grid container
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.info_panel)
        self.splitter.addWidget(self.grid_container)

        # Disable splitter's built-in collapse functionality
        self.splitter.setChildrenCollapsible(False)

        # Set stretch factors: info panel fixed, grid container expands
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        assert event is not None
        if event.key() == QtCore.Qt.Key.Key_R:
            # Reset zoom on active stream
            if self.streams:
                self.streams[self.active_stream_index].view.reset_zoom()
        else:
            super().keyPressEvent(event)
