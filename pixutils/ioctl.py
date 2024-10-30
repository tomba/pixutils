from ctypes import sizeof

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(iodir, iotype, nr, size):  # pylint: disable=invalid-name
    return (((iodir << _IOC_DIRSHIFT) | (ord(iotype) << _IOC_TYPESHIFT)) | (nr << _IOC_NRSHIFT)) | (
        size << _IOC_SIZESHIFT
    )


def IO(iotype, nr):  # pylint: disable=invalid-name
    return _IOC(_IOC_NONE, iotype, nr, 0)


def IOR(iotype, nr, size):  # pylint: disable=invalid-name
    return _IOC(_IOC_READ, iotype, nr, sizeof(size))


def IOW(iotype, nr, size):  # pylint: disable=invalid-name
    return _IOC(_IOC_WRITE, iotype, nr, sizeof(size))


def IOWR(iotype, nr, size):  # pylint: disable=invalid-name
    return _IOC((_IOC_READ | _IOC_WRITE), iotype, nr, sizeof(size))
