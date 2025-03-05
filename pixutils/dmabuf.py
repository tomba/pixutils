import ctypes
import fcntl

from enum import IntFlag

from pixutils.ioctl import IOW

__all__ = ['DmaBufSyncFlags', 'dmabuf_sync']


# pylint: disable=invalid-name
class struct_dma_buf_sync(ctypes.Structure):
    __slots__ = ['flags']
    _fields_ = [('flags', ctypes.c_uint64)]


class DmaBufSyncFlags(IntFlag):
    READ = 1 << 0
    WRITE = 2 << 0
    RW = READ | WRITE
    START = 0 << 2
    END = 1 << 2


DMA_BUF_BASE = 'b'
DMA_BUF_IOCTL_SYNC = IOW(DMA_BUF_BASE, 0, struct_dma_buf_sync)


def dmabuf_sync(fd: int, flags: DmaBufSyncFlags):
    req = struct_dma_buf_sync()
    req.flags = flags
    fcntl.ioctl(fd, DMA_BUF_IOCTL_SYNC, req, True)
