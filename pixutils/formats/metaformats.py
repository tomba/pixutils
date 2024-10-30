from __future__ import annotations

from .fourcc_str import str_to_fourcc

__all__ = ['MetaFormat', 'MetaFormats']


class MetaFormat:
    def __init__(
        self, name: str, v4l2_fourcc: str, pixelspergroup: int, bytespergroup: int
    ) -> None:
        self.name = name
        self.v4l2_fourcc = str_to_fourcc(v4l2_fourcc)
        self.pixelspergroup = pixelspergroup
        self.bytespergroup = bytespergroup

    def stride(self, width: int, align: int = 1):
        # ceil(width / pixelsPerGroup) * bytesPerGroup
        stride = (width + self.pixelspergroup - 1) // self.pixelspergroup * self.bytespergroup

        # ceil(stride / align) * align
        return (stride + align - 1) // align * align

    def buffersize(self, width: int, height: int, align: int = 1):
        stride = self.stride(width, align)
        if stride == 0:
            return 0

        return stride * height


class MetaFormats:
    __FMT_LIST: list[MetaFormat] = []

    @staticmethod
    def __init_fmt_list():
        # Perhaps there is some better way to handle this...
        if not MetaFormats.__FMT_LIST:
            MetaFormats.__FMT_LIST = [
                v for v in MetaFormats.__dict__.values() if isinstance(v, MetaFormat)
            ]

    @staticmethod
    def find_v4l2_fourcc(fourcc):
        MetaFormats.__init_fmt_list()
        return next(f for f in MetaFormats.__FMT_LIST if f.v4l2_fourcc == fourcc)

    @staticmethod
    def find_by_name(name):
        MetaFormats.__init_fmt_list()
        return next(f for f in MetaFormats.__FMT_LIST if f.name == name)

    GENERIC_8 = MetaFormat('GENERIC_8', 'MET8', 2, 2)
    GENERIC_CSI2_10 = MetaFormat('GENERIC_CSI2_10', 'MC1A', 4, 5)
    GENERIC_CSI2_12 = MetaFormat('GENERIC_CSI2_12', 'MC1C', 2, 3)

    RPI_FE_CFG = MetaFormat('RPI_FE_CFG', 'RPFC', 1, 1)
    RPI_FE_STATS = MetaFormat('RPI_FE_STATS', 'RPFS', 1, 1)
