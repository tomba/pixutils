from __future__ import annotations

from .fourcc_str import str_to_fourcc, fourcc_to_str

__all__ = [ 'MetaFormat', 'MetaFormats' ]


class MetaFormat:
    def __init__(self, name: str, v4l2_fourcc: str, pixelspergroup: int, bytespergroup: int) -> None:
        self.name = name
        self.v4l2_fourcc = str_to_fourcc(v4l2_fourcc)
        self.pixelspergroup = pixelspergroup
        self.bytespergroup = bytespergroup

    def stride(self, width: int, align: int = 1):
        # ceil(width / pixelsPerGroup) * bytesPerGroup
        stride = (width + self.pixelspergroup - 1) // self.pixelspergroup * self.bytespergroup

        # ceil(stride / align) * align
        return (stride + align - 1) // align * align

    def buffersize(self, width, height, align: int = 1):
        stride = self.stride(width, align)
        if stride == 0:
            return 0

        return stride * height


class MetaFormats:
    @staticmethod
    def find_v4l2_fourcc(fourcc):
        return next(v for v in MetaFormats.__dict__.values() if isinstance(v, MetaFormat) and v.v4l2_fourcc == fourcc)

    @staticmethod
    def find_v4l2_fourcc_unsupported(fourcc):
        try:
            return MetaFormats.find_v4l2_fourcc(fourcc)
        except StopIteration:
            s = fourcc_to_str(fourcc)
            return MetaFormat(f'Unsupported<{s}>', s, 0, 0)

    @staticmethod
    def find_by_name(name):
        return next(v for v in MetaFormats.__dict__.values() if isinstance(v, MetaFormat) and v.name == name)

    GENERIC_8 = MetaFormat('GENERIC_8', 'MET8', 2, 2)
    GENERIC_CSI2_10 = MetaFormat('GENERIC_CSI2_10', 'MC1A', 4, 5)
    GENERIC_CSI2_12 = MetaFormat('GENERIC_CSI2_12', 'MC1C', 2, 3)

    RPI_FE_CFG = MetaFormat('RPI_FE_CFG', 'RPFC', 1, 1)
    RPI_FE_STATS = MetaFormat('RPI_FE_STATS', 'RPFS', 1, 1)
