from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from .fourcc_str import str_to_fourcc

__all__ = ['PixelColorEncoding', 'PixelFormat', 'PixelFormats']


class PixelColorEncoding(Enum):
    RGB = 0
    YUV = 1
    RAW = 2
    UNDEFINED = 3


class PixelFormatPlaneInfo(NamedTuple):
    bytespergroup: int
    verticalsubsampling: int


class PixelFormat:
    def __init__(
        self,
        name: str,
        drm_fourcc: None | str,
        v4l2_fourcc: None | str,
        colorencoding: PixelColorEncoding,
        packed: bool,
        pixelspergroup: int,
        planes,
    ) -> None:
        self.name = name
        self.drm_fourcc = str_to_fourcc(drm_fourcc) if drm_fourcc else None
        self.v4l2_fourcc = str_to_fourcc(v4l2_fourcc) if v4l2_fourcc else None
        self.color = colorencoding
        self.packed = packed
        self.pixelspergroup = pixelspergroup
        self.planes = [PixelFormatPlaneInfo(*p) for p in planes]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'PixelFormat({self.name})'

    def stride(self, width: int, plane: int = 0, align: int = 1):
        if plane >= len(self.planes):
            raise RuntimeError()

        # ceil(width / pixelsPerGroup) * bytesPerGroup
        stride = (
            (width + self.pixelspergroup - 1)
            // self.pixelspergroup
            * self.planes[plane].bytespergroup
        )

        # ceil(stride / align) * align
        return (stride + align - 1) // align * align

    def planesize(self, width, height, plane: int = 0, align: int = 1):
        stride = self.stride(width, plane, align)
        if stride == 0:
            return 0

        # return self.planeSize(height, plane, stride)

        vertsubsample = self.planes[plane].verticalsubsampling

        # stride * ceil(height / verticalSubSampling)
        return stride * ((height + vertsubsample - 1) // vertsubsample)

    #    def planesize(self, height, plane, stride):
    #        vertSubSample = self.planes[plane].verticalSubSampling
    #        if vertSubSample == 0:
    #            return 0
    #
    #        # stride * ceil(height / verticalSubSampling)
    #        return stride * ((height + vertSubSample - 1) // vertSubSample)

    def framesize(self, width, height, align: int = 1):
        return sum(self.planesize(width, height, i, align) for i in range(len(self.planes)))


class PixelFormats:
    __FMT_LIST: list[PixelFormat] = []

    @staticmethod
    def __init_fmt_list():
        # Perhaps there is some better way to handle this...
        if not PixelFormats.__FMT_LIST:
            PixelFormats.__FMT_LIST = [
                v for v in PixelFormats.__dict__.values() if isinstance(v, PixelFormat)
            ]

    @staticmethod
    def find_v4l2_fourcc(fourcc: int):
        PixelFormats.__init_fmt_list()
        return next(f for f in PixelFormats.__FMT_LIST if f.v4l2_fourcc == fourcc)

    @staticmethod
    def find_drm_fourcc(fourcc: int):
        PixelFormats.__init_fmt_list()
        return next(f for f in PixelFormats.__FMT_LIST if f.drm_fourcc == fourcc)

    @staticmethod
    def find_by_name(name):
        PixelFormats.__init_fmt_list()
        return next(f for f in PixelFormats.__FMT_LIST if f.name == name)

    @staticmethod
    def get_formats():
        PixelFormats.__init_fmt_list()
        return PixelFormats.__FMT_LIST

    # RGB 16-bit, no alpha

    RGB565 = PixelFormat(
        'RGB565',
        'RG16',
        'RGBP',
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )
    XRGB1555 = PixelFormat(
        'XRGB1555',
        'XR15',  # DRM_FORMAT_XRGB1555
        None,
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )
    RGBX4444 = PixelFormat(
        'RGBX4444',
        'RX12',  # DRM_FORMAT_RGBX4444
        None,
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )
    XRGB4444 = PixelFormat(
        'XRGB4444',
        'XR12',  # DRM_FORMAT_XRGB4444
        None,
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )

    # RGB 16-bit, alpha

    ARGB1555 = PixelFormat(
        'ARGB1555',
        'AR15',  # DRM_FORMAT_ARGB1555
        None,
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )
    RGBA4444 = PixelFormat(
        'RGBA4444',
        'RA12',  # DRM_FORMAT_RGBA4444
        None,
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )
    ARGB4444 = PixelFormat(
        'ARGB4444',
        'AR12',  # DRM_FORMAT_ARGB4444
        None,
        PixelColorEncoding.RGB,
        False,
        1,
        ((2, 1),),
    )

    # RGB 24-bit

    RGB888 = PixelFormat(
        'RGB888',
        'RG24',  # DRM_FORMAT_RGB888
        'BGR3',  # V4L2_PIX_FMT_BGR24
        PixelColorEncoding.RGB,
        False,
        1,
        ((3, 1),),
    )
    BGR888 = PixelFormat(
        'BGR888',
        'BG24',  # DRM_FORMAT_BGR888
        'RGB3',  # V4L2_PIX_FMT_RGB24
        PixelColorEncoding.RGB,
        False,
        1,
        ((3, 1),),
    )

    # RGB 32-bit, no alpha

    XRGB8888 = PixelFormat(
        'XRGB8888',
        'XR24',  # DRM_FORMAT_XRGB8888
        'XR24',  # V4L2_PIX_FMT_XBGR32
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )
    XBGR8888 = PixelFormat(
        'XBGR8888',
        'XB24',  # DRM_FORMAT_XBGR8888
        'XB24',  # V4L2_PIX_FMT_RGBX32
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )
    RGBX8888 = PixelFormat(
        'RGBX8888',
        'RX24',  # DRM_FORMAT_RGBX8888
        'RX24',  # V4L2_PIX_FMT_BGRX32
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )
    XBGR2101010 = PixelFormat(
        'XBGR2101010',
        'XB30',  # DRM_FORMAT_XBGR2101010
        'RX30',  # V4L2_PIX_FMT_RGBX1010102
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )

    # RGB 32-bit, alpha

    ARGB8888 = PixelFormat(
        'ARGB8888',
        'AR24',  # DRM_FORMAT_ARGB8888
        'AR24',  # V4L2_PIX_FMT_ABGR32
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )
    ABGR8888 = PixelFormat(
        'ABGR8888',
        'AB24',  # DRM_FORMAT_ABGR8888
        'AB24',  # V4L2_PIX_FMT_RGBA32
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )
    RGBA8888 = PixelFormat(
        'RGBA8888',
        'RA24',  # DRM_FORMAT_RGBA8888
        'RA24',  # V4L2_PIX_FMT_BGRA32
        PixelColorEncoding.RGB,
        False,
        1,
        ((4, 1),),
    )

    # YUV

    NV12 = PixelFormat(
        'NV12',
        'NV12',
        'NM12',
        PixelColorEncoding.YUV,
        False,
        2,
        (
            (2, 1),
            (2, 2),
        ),
    )

    NV16 = PixelFormat(
        'NV16',
        'NV16',
        'NM16',
        PixelColorEncoding.YUV,
        False,
        2,
        (
            (2, 1),
            (2, 1),
        ),
    )

    YUYV = PixelFormat(
        'YUYV',
        'YUYV',
        'YUYV',
        PixelColorEncoding.YUV,
        False,
        2,
        ((4, 1),),
    )

    UYVY = PixelFormat(
        'UYVY',
        'UYVY',
        'UYVY',
        PixelColorEncoding.YUV,
        False,
        2,
        ((4, 1),),
    )

    # YUV 4:4:4

    VUY888 = PixelFormat(
        'VUY888',
        'VU24',  # DRM_FORMAT_VUY888
        'YUV3',  # V4L2_PIX_FMT_YUV24
        PixelColorEncoding.YUV,
        False,
        1,
        ((3, 1),),
    )

    XVUY8888 = PixelFormat(
        'XVUY8888',
        'XVUY',  # DRM_FORMAT_XVUY8888
        'YUVX',  # V4L2_PIX_FMT_YUVX32
        PixelColorEncoding.YUV,
        False,
        1,
        ((4, 1),),
    )

    # Y8

    Y8 = PixelFormat(
        'Y8',
        None,
        'GREY',
        PixelColorEncoding.YUV,
        False,
        1,
        ((1, 1),),
    )

    # RAW

    SBGGR8 = PixelFormat(
        'SBGGR8',
        None,
        'BA81',
        PixelColorEncoding.RAW,
        False,
        2,
        ((2, 1),),
    )

    SGBRG8 = PixelFormat(
        'SGBRG8',
        None,
        'GBRG',
        PixelColorEncoding.RAW,
        False,
        2,
        ((2, 1),),
    )

    SGRBG8 = PixelFormat(
        'SGRBG8',
        None,
        'GRBG',
        PixelColorEncoding.RAW,
        False,
        2,
        ((2, 1),),
    )

    SRGGB8 = PixelFormat(
        'SRGGB8',
        None,
        'RGGB',
        PixelColorEncoding.RAW,
        False,
        2,
        ((2, 1),),
    )

    SRGGB10 = PixelFormat(
        'SRGGB10',
        None,
        'RG10',
        PixelColorEncoding.RAW,
        False,
        2,
        ((4, 1),),
    )

    SBGGR10 = PixelFormat(
        'SBGGR10',
        None,
        'BG10',
        PixelColorEncoding.RAW,
        False,
        2,
        ((4, 1),),
    )

    SRGGB10P = PixelFormat(
        'SRGGB10P',
        None,
        'pRAA',
        PixelColorEncoding.RAW,
        True,
        4,
        ((5, 1),),
    )

    SRGGB12 = PixelFormat(
        'SRGGB12',
        None,
        'RG12',
        PixelColorEncoding.RAW,
        False,
        2,
        ((4, 1),),
    )

    SRGGB16 = PixelFormat(
        'SRGGB16',
        None,
        'RG16',
        PixelColorEncoding.RAW,
        False,
        2,
        ((4, 1),),
    )

    # Compressed formats
    MJPEG = PixelFormat(
        'MJPEG',
        'MJPG',
        'MJPG',
        PixelColorEncoding.YUV,
        False,
        1,
        ((1, 1),),
    )
