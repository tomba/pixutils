from __future__ import annotations

from enum import Enum
from math import ceil
from typing import NamedTuple

from .fourcc_str import fourcc_to_str, str_to_fourcc

__all__ = ['PixelColorEncoding', 'PixelFormat', 'PixelFormats']


class PixelColorEncoding(Enum):
    RGB = 0
    YUV = 1
    RAW = 2
    UNDEFINED = 3


class PixelFormatPlaneInfo(NamedTuple):
    bytes_per_block: int
    pixels_per_block: int
    hsub: int
    vsub: int

def _div_round_up(a: int, b: int):
    return int(ceil(a / b))

def _align_up(a: int, b: int):
    return _div_round_up(a, b) * b

class PixelFormat:
    def __init__(self, name: str,
                 drm_fourcc: None | str, v4l2_fourcc: None | str,
                 colorencoding: PixelColorEncoding, packed: bool,
                 pixel_align: tuple[int, int], planes) -> None:
        self.name = name
        self.drm_fourcc = str_to_fourcc(drm_fourcc) if drm_fourcc else None
        self.v4l2_fourcc = str_to_fourcc(v4l2_fourcc) if v4l2_fourcc else None
        self.color = colorencoding
        self.packed = packed
        # pixel alignment (width, height)
        self.pixel_align = pixel_align

        def adjust_p(p):
            if len(p) == 1:
                return (p[0], 1, 1, 1)
            if len(p) == 2:
                return (p[0], p[1], 1, 1)

            assert len(p) == 4

            return p

        self.planes = [PixelFormatPlaneInfo(*adjust_p(p)) for p in planes]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'PixelFormat({self.name})'

    def align_pixels(self, width: int, height: int):
        return (_align_up(width, self.pixel_align[0]),
                _align_up(height, self.pixel_align[1]))

    def stride(self, width: int, plane: int = 0, align = 1):
        if plane >= len(self.planes):
            raise RuntimeError()

        assert width % self.pixel_align[0] == 0

        pi = self.planes[plane]

        assert width % pi.pixels_per_block == 0
        stride = width // pi.pixels_per_block * pi.bytes_per_block

        assert stride % pi.hsub == 0
        stride = stride // pi.hsub

        stride = _align_up(stride, align)

        return stride

    def planesize(self, stride: int, height: int, plane: int = 0):
        assert height % self.pixel_align[1] == 0

        pi = self.planes[plane]

        assert height % pi.vsub == 0

        return stride * (height // pi.vsub)


    def framesize(self, width: int, height: int, align = 1):
        size = 0

        for i in range(len(self.planes)):
            stride = self.stride(width, i, align)
            size += self.planesize(stride, height, i)

        return size

    def dumb_size(self, width: int, height: int, plane: int = 0, align = 1):
        """
        Helper function mainly for DRM dumb framebuffer
        Returns (width, height, bitspp) tuple which results in a suitable plane
        size.

        DRM_IOCTL_MODE_CREATE_DUMB takes a 'bpp' (bits-per-pixel) argument,
        which is then used with the width and height to allocate the buffer.
        This doesn't work for pixel formats where the average bits-per-pixel
        is not an integer (e.g. XV15)

        So, we instead use the bytes_per_block (in bits) as
        the 'bpp' argument, and adjust the width accordingly.
        """

        assert height % self.pixel_align[1] == 0

        pi = self.planes[plane]

        assert height % pi.vsub == 0

        stride = self.stride(width, plane, align)

        assert stride % pi.bytes_per_block == 0

        width = stride // pi.bytes_per_block
        height = height // pi.vsub
        bitspp = pi.bytes_per_block * 8

        return width, height, bitspp


class PixelFormats:
    __FMT_LIST: list[PixelFormat] = []

    @staticmethod
    def __init_fmt_list():
        # Perhaps there is some better way to handle this...
        if not PixelFormats.__FMT_LIST:
            PixelFormats.__FMT_LIST = [v for v in PixelFormats.__dict__.values() if isinstance(v, PixelFormat)]

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

    # Single 8-bit channel
    R8 = PixelFormat('R8',
        'R8  ', None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 1, ), ),
    )

    # RGB 8-bit
    RGB332 = PixelFormat('RGB332',
        'RGB8', None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 1, ), ),
    )

    # RGB 16-bit, no alpha

    RGB565 = PixelFormat('RGB565',
        'RG16', 'RGBP',
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )
    BGR565 = PixelFormat('BGR565',
        'BG16', None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )
    XRGB1555 = PixelFormat('XRGB1555',
        'XR15',     # DRM_FORMAT_XRGB1555
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )
    RGBX4444 = PixelFormat('RGBX4444',
        'RX12',     # DRM_FORMAT_RGBX4444
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )
    XRGB4444 = PixelFormat('XRGB4444',
        'XR12',     # DRM_FORMAT_XRGB4444
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )

    # RGB 16-bit, alpha

    ARGB1555 = PixelFormat('ARGB1555',
        'AR15',     # DRM_FORMAT_ARGB1555
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )
    RGBA4444 = PixelFormat('RGBA4444',
        'RA12',     # DRM_FORMAT_RGBA4444
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )
    ARGB4444 = PixelFormat('ARGB4444',
        'AR12',     # DRM_FORMAT_ARGB4444
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 2, ), ),
    )

    # RGB 24-bit

    RGB888 = PixelFormat('RGB888',
        'RG24',     # DRM_FORMAT_RGB888
        'BGR3',     # V4L2_PIX_FMT_BGR24
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 3, ), ),
    )
    BGR888 = PixelFormat('BGR888',
        'BG24',     # DRM_FORMAT_BGR888
        'RGB3',     # V4L2_PIX_FMT_RGB24
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 3, ), ),
    )

    # RGB 32-bit, no alpha

    XRGB8888 = PixelFormat('XRGB8888',
        'XR24',     # DRM_FORMAT_XRGB8888
        'XR24',     # V4L2_PIX_FMT_XBGR32
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    XBGR8888 = PixelFormat('XBGR8888',
        'XB24',     # DRM_FORMAT_XBGR8888
        'XB24',     # V4L2_PIX_FMT_RGBX32
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    RGBX8888 = PixelFormat('RGBX8888',
        'RX24',     # DRM_FORMAT_RGBX8888
        'RX24',     # V4L2_PIX_FMT_BGRX32
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    BGRX8888 = PixelFormat('BGRX8888',
        'BX24',     # DRM_FORMAT_RGBX8888
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )

    XBGR2101010 = PixelFormat('XBGR2101010',
        'XB30',     # DRM_FORMAT_XBGR2101010
        'RX30',     # V4L2_PIX_FMT_RGBX1010102
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    XRGB2101010 = PixelFormat('XRGB2101010',
        'XR30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    RGBX1010102 = PixelFormat('RGBX1010102',
        'RX30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    BGRX1010102 = PixelFormat('BGRX1010102',
        'BX30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )

    # RGB 32-bit, alpha

    ARGB8888 = PixelFormat('ARGB8888',
        'AR24',     # DRM_FORMAT_ARGB8888
        'AR24',     # V4L2_PIX_FMT_ABGR32
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    ABGR8888 = PixelFormat('ABGR8888',
        'AB24',     # DRM_FORMAT_ABGR8888
        'AB24',     # V4L2_PIX_FMT_RGBA32
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    RGBA8888 = PixelFormat('RGBA8888',
        'RA24',     # DRM_FORMAT_RGBA8888
        'RA24',     # V4L2_PIX_FMT_BGRA32
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    BGRA8888 = PixelFormat('BGRA8888',
        'BA24',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )


    ABGR2101010 = PixelFormat('ABGR2101010',
        'AB30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    ARGB2101010 = PixelFormat('ARGB2101010',
        'AR30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    RGBA1010102 = PixelFormat('RGBA1010102',
        'RA30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )
    BGRA1010102 = PixelFormat('BGRA1010102',
        'BA30',
        None,
        PixelColorEncoding.RGB,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )

    # YUV Packed

    YUYV = PixelFormat('YUYV',
        'YUYV', 'YUYV',
        PixelColorEncoding.YUV,
        False,
        ( 2, 1 ),
        ( ( 4, 2, 1, 1 ), ),
    )

    UYVY = PixelFormat('UYVY',
        'UYVY', 'UYVY',
        PixelColorEncoding.YUV,
        False,
        ( 2, 1 ),
        ( ( 4, 2, 1, 1 ), ),
    )

    YVYU = PixelFormat('YVYU',
        'YVYU', 'YVYU',
        PixelColorEncoding.YUV,
        False,
        ( 2, 1 ),
        ( ( 4, 2, 1, 1 ), ),
    )

    VYUY = PixelFormat('VYUY',
        'VYUY', 'VYUY',
        PixelColorEncoding.YUV,
        False,
        ( 2, 1 ),
        ( ( 4, 2, 1, 1 ), ),
    )

    VUY888 = PixelFormat('VUY888',
        'VU24',     # DRM_FORMAT_VUY888
        'YUV3',     # V4L2_PIX_FMT_YUV24
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 3, ), ),
    )

    XVUY8888 = PixelFormat('XVUY8888',
        'XVUY',     # DRM_FORMAT_XVUY8888
        'YUVX',     # V4L2_PIX_FMT_YUVX32
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 4, ), ),
    )

    Y210 = PixelFormat('Y210',
        'Y210',     # DRM_FORMAT_Y210
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 8, 2, 1, 1 ), ),
    )

    Y212 = PixelFormat('Y212',
        'Y212',     # DRM_FORMAT_Y212
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 8, 2, 1, 1 ), ),
    )

    Y216 = PixelFormat('Y216',
        'Y216',     # DRM_FORMAT_Y216
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 8, 2, 1, 1 ), ),
    )

    # YUV Semi Planar

    NV12 = PixelFormat('NV12',
        'NV12', 'NM12',
        PixelColorEncoding.YUV,
        False,
        ( 2, 2 ),
        ( ( 1, 1, 1, 1 ),
          ( 2, 1, 2, 2), ),
    )

    NV21 = PixelFormat('NV21',
        'NV21', 'NM21',
        PixelColorEncoding.YUV,
        False,
        ( 2, 2 ),
        ( ( 1, 1, 1, 1 ),
          ( 2, 1, 2, 2), ),
    )

    NV16 = PixelFormat('NV16',
        'NV16', 'NM16',
        PixelColorEncoding.YUV,
        False,
        ( 2, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 2, 1, 2, 1), ),
    )

    NV61 = PixelFormat('NV61',
        'NV61', 'NM61',
        PixelColorEncoding.YUV,
        False,
        ( 2, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 2, 1, 2, 1), ),
    )

    XV15 = PixelFormat('XV15',
        'XV15', None,
        PixelColorEncoding.YUV,
        False,
        (6, 2),
        ( ( 4, 3, 1, 1, ),
          ( 8, 3, 2, 2, ), ),
    )

    XV20 = PixelFormat('XV20',
        'XV20', None,
        PixelColorEncoding.YUV,
        False,
        (6, 2),
        ( ( 4, 3, 1, 1, ),
          ( 8, 3, 2, 1, ), ),
    )

    XVUY2101010 = PixelFormat('XVUY2101010',
        'XY30', None,
        PixelColorEncoding.YUV,
        False,
        (1, 1),
        ( ( 4, ), ),
    )

    # YUV Planar

    YUV420 = PixelFormat('YUV420',
        'YU12',
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 1, 1, 2, 2 ),
          ( 1, 1, 2, 2 ), ),
    )

    YVU420 = PixelFormat('YVU420',
        'YV12',
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 1, 1, 2, 2 ),
          ( 1, 1, 2, 2 ), ),
    )

    YUV422 = PixelFormat('YUV422',
        'YU16',
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 1, 1, 2, 1 ),
          ( 1, 1, 2, 1 ), ),
    )

    YVU422 = PixelFormat('YVU422',
        'YV16',
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 1, 1, 2, 1 ),
          ( 1, 1, 2, 1 ), ),
    )

    YUV444 = PixelFormat('YUV444',
        'YU24',
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 1, 1, 1, 1 ),
          ( 1, 1, 1, 1 ), ),
    )

    YVU444 = PixelFormat('YVU444',
        'YV24',
        None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, 1, 1, 1 ),
          ( 1, 1, 1, 1 ),
          ( 1, 1, 1, 1 ), ),
    )

    X403 = PixelFormat('X403',
        'X403', None,
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 4, ),
          ( 4, ),
          ( 4, ) ),
    )

    # Grey formats

    Y8 = PixelFormat('Y8',
        'GREY', 'GREY',
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, ), ),
    )

    Y10 = PixelFormat('Y10',
        None, 'Y10 ',
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 2, 1 ), ),
    )

    Y10P = PixelFormat('Y10P',
        None, 'Y10P',
        PixelColorEncoding.YUV,
        True,
        ( 4, 1 ),
        ( ( 5, 4 ), ),
    )

    Y12 = PixelFormat('Y12',
        None, 'Y12 ',
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 2, 1 ), ),
    )

    Y12P = PixelFormat('Y12P',
        None, 'Y12P',
        PixelColorEncoding.YUV,
        True,
        ( 2, 1 ),
        ( ( 3, 2 ), ),
    )

    Y10_LE32 = PixelFormat('Y10_P32',
        'YPA4', None,
        PixelColorEncoding.YUV,
        False,
        ( 3, 1 ),
        ( ( 4, 3, ), ),
    )

    # RAW Bayer formats

    SBGGR8 = PixelFormat('SBGGR8',
        None, 'BA81',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 1, 1 ), ),
    )

    SGBRG8 = PixelFormat('SGBRG8',
        None, 'GBRG',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 1, 1 ), ),
    )

    SGRBG8 = PixelFormat('SGRBG8',
        None, 'GRBG',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 1, 1 ), ),
    )

    SRGGB8 = PixelFormat('SRGGB8',
        None, 'RGGB',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 1, 1 ), ),
    )

    SBGGR10 = PixelFormat('SBGGR10',
        None, 'BG10',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SGBRG10 = PixelFormat('SGBRG10',
        None, 'GB10',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SGRBG10 = PixelFormat('SGRBG10',
        None, 'BA10',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SRGGB10 = PixelFormat('SRGGB10',
        None, 'RG10',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SBGGR10P = PixelFormat('SBGGR10P',
        None, 'pBAA',
        PixelColorEncoding.RAW,
        True,
        ( 4, 2 ),
        ( ( 5, 4 ), ),
    )

    SGBRG10P = PixelFormat('SGBRG10P',
        None, 'pGAA',
        PixelColorEncoding.RAW,
        True,
        ( 4, 2 ),
        ( ( 5, 4 ), ),
    )

    SGRBG10P = PixelFormat('SGRBG10P',
        None, 'pgAA',
        PixelColorEncoding.RAW,
        True,
        ( 4, 2 ),
        ( ( 5, 4 ), ),
    )

    SRGGB10P = PixelFormat('SRGGB10P',
        None, 'pRAA',
        PixelColorEncoding.RAW,
        True,
        ( 4, 2 ),
        ( ( 5, 4 ), ),
    )

    SBGGR12 = PixelFormat('SBGGR12',
        None, 'BG12',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SGBRG12 = PixelFormat('SGBRG12',
        None, 'GB12',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SGRBG12 = PixelFormat('SGRBG12',
        None, 'BA12',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SRGGB12 = PixelFormat('SRGGB12',
        None, 'RG12',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SBGGR12P = PixelFormat('SBGGR12P',
        None, 'pBCC',
        PixelColorEncoding.RAW,
        True,
        ( 2, 2 ),
        ( ( 3, 2 ), ),
    )

    SGBRG12P = PixelFormat('SGBRG12P',
        None, 'pGCC',
        PixelColorEncoding.RAW,
        True,
        ( 2, 2 ),
        ( ( 3, 2 ), ),
    )

    SGRBG12P = PixelFormat('SGRBG12P',
        None, 'pgCC',
        PixelColorEncoding.RAW,
        True,
        ( 2, 2 ),
        ( ( 3, 2 ), ),
    )

    SRGGB12P = PixelFormat('SRGGB12P',
        None, 'pRCC',
        PixelColorEncoding.RAW,
        True,
        ( 2, 2 ),
        ( ( 3, 2 ), ),
    )

    SBGGR16 = PixelFormat('SBGGR16',
        None, 'BYR2',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SGBRG16 = PixelFormat('SGBRG16',
        None, 'GB16',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SGRBG16 = PixelFormat('SGRBG16',
        None, 'GR16',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    SRGGB16 = PixelFormat('SRGGB16',
        None, 'RG16',
        PixelColorEncoding.RAW,
        False,
        ( 2, 2 ),
        ( ( 4, 2 ), ),
    )

    # Compressed formats
    MJPEG = PixelFormat('MJPEG',
        'MJPG', 'MJPG',
        PixelColorEncoding.YUV,
        False,
        ( 1, 1 ),
        ( ( 1, ), ),
    )

# Helper to dump the pixel formats into a C++ struct
def dump_c_structs():
    for fmt in PixelFormats.get_formats():
        print(f'\t{fmt.name},')

    print()
    print()

    print('{')

    for fmt in PixelFormats.get_formats():
        print('\t{')
        print(f'\t\tPixelFormat::{fmt.name}, {{')
        print('\t\t\tPixelFormatInfo {')
        print(f'\t\t\t\t"{fmt.name}",')
        print(f'\t\t\t\t"{fourcc_to_str(fmt.drm_fourcc) if fmt.drm_fourcc else ""}",')
        print(f'\t\t\t\t"{fourcc_to_str(fmt.v4l2_fourcc) if fmt.v4l2_fourcc else ""}",')
        print(f'\t\t\t\tPixelColorType::{fmt.color.name},')
        print(f'\t\t\t\t{{ {fmt.pixel_align[0]}, {fmt.pixel_align[1]} }},')

        planedata = [f'{{ {p.bytes_per_block}, {p.pixels_per_block}, {p.hsub}, {p.vsub} }}' for p in fmt.planes]

        print(f'\t\t\t\t{{ {", ".join(planedata)} }},')

        print('\t\t\t}')
        print('\t\t}')
        print('\t},')

    print('}')

# Validate that the format names match the field names, and that the fourccs are unique
def validate_formats():
    for f in PixelFormats.get_formats():
        # pylint: disable=consider-iterating-dictionary
        assert f.name in PixelFormats.__dict__.keys(), f.name

    names = [f.name for f in PixelFormats.get_formats()]
    assert len(names) == len(set(names))

    drm_fourccs = [f.drm_fourcc for f in PixelFormats.get_formats() if f.drm_fourcc]
    assert len(drm_fourccs) == len(set(drm_fourccs))

    v4l2_fourccs = [f.v4l2_fourcc for f in PixelFormats.get_formats() if f.v4l2_fourcc]
    assert len(v4l2_fourccs) == len(set(v4l2_fourccs))
