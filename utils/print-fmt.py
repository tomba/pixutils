#!/usr/bin/python3

import argparse

from pixutils.formats import PixelFormats, fourcc_to_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('format')
    parser.add_argument('-a', action='store_true', help='align pixel size')
    parser.add_argument('width', nargs='?', default=640, type=int)
    parser.add_argument('height', nargs='?', default=480, type=int)
    parser.add_argument('align', nargs='?', default=1, type=int, help='stride byte align')
    args = parser.parse_args()

    width = args.width
    height = args.height
    align = args.align

    try:
        fmt = PixelFormats.find_by_name(args.format)
    except StopIteration:
        print(f'Format "{args.format}" not found')
        return -1

    if fmt.drm_fourcc:
        drm_fourcc = fourcc_to_str(fmt.drm_fourcc)
    else:
        drm_fourcc = '    '

    if fmt.v4l2_fourcc:
        v4l2_fourcc = fourcc_to_str(fmt.v4l2_fourcc)
    else:
        v4l2_fourcc = '    '

    if args.a:
        width,height = fmt.align_pixels(width, height)

    print(f'{fmt.name}')
    print(f'drm_fourcc="{drm_fourcc:4}" v4l2_fourcc="{v4l2_fourcc:4}"')
    print(f'color={fmt.color.name} pixel_align={fmt.pixel_align}')

    print(f'width={width} height={height} align={align}')

    for pi,p in enumerate(fmt.planes):
        stride = fmt.stride(width, pi, align)
        psize = fmt.planesize(stride, height, pi)
        dsize = fmt.dumb_size(width, height, pi, align)

        assert psize == dsize[0] * dsize[1] * dsize[2] / 8

        print(f'plane{pi} bytes_per_block={p.bytes_per_block} pixels_per_block={p.pixels_per_block} hsub={p.hsub} vsub={p.vsub}')
        print(f'       stride={stride} plane_size={psize} dumb_size={dsize}')

    print(f'framesize={fmt.framesize(width, height, align)}')

    return 0


if __name__ == '__main__':
    main()
