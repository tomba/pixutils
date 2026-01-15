from __future__ import annotations

__all__ = ['fourcc_to_str', 'str_to_fourcc']


def fourcc_to_str(fourcc: int):
    return ''.join(
        (
            chr((fourcc >> 0) & 0xFF),
            chr((fourcc >> 8) & 0xFF),
            chr((fourcc >> 16) & 0xFF),
            chr((fourcc >> 24) & 0xFF),
        )
    )


def str_to_fourcc(s: str):
    if len(s) != 4:
        raise ValueError('Invalid fourcc string')

    return ord(s[0]) << 0 | ord(s[1]) << 8 | ord(s[2]) << 16 | ord(s[3]) << 24
