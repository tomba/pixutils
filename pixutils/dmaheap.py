import ctypes

import fcntl
import os
import weakref

from pixutils.ioctl import IOWR

__all__ = ['DMAHeap', 'DMAHeapBuffer']

# pylint: disable=invalid-name
class struct_dma_heap_allocation_data(ctypes.Structure):
    __slots__ = ['len', 'fd', 'fd_flags', 'heap_flags']
    _fields_ = [('len', ctypes.c_uint64),
                ('fd', ctypes.c_uint32),
                ('fd_flags', ctypes.c_uint32),
                ('heap_flags', ctypes.c_uint64)]

DMA_HEAP_IOC_MAGIC = 'H'

DMA_HEAP_IOCTL_ALLOC = IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct_dma_heap_allocation_data)


class DMAHeap:
    def __init__(self, name: str):
        self.fd = os.open(f'/dev/dma_heap/{name}', os.O_CLOEXEC | os.O_RDWR)

        weakref.finalize(self, os.close, self.fd)

    def alloc(self, length: int):
        # pylint: disable=attribute-defined-outside-init
        buf_data = struct_dma_heap_allocation_data()
        buf_data.len = length
        buf_data.fd_flags = os.O_CLOEXEC | os.O_RDWR
        fcntl.ioctl(self.fd, DMA_HEAP_IOCTL_ALLOC, buf_data, True)

        return DMAHeapBuffer(buf_data.fd, buf_data.len)


class DMAHeapBuffer:
    def __init__(self, fd: int, length: int):
        self.fd = fd
        self.length = length

        weakref.finalize(self, os.close, self.fd)
