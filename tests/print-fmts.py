#!/usr/bin/python3

from pixutils.formats import PixelFormats, fourcc_to_str

def main():
    for fmt in PixelFormats.get_formats():
        if fmt.drm_fourcc:
            drm_fourcc = fourcc_to_str(fmt.drm_fourcc)
        else:
            drm_fourcc = '    '

        if fmt.v4l2_fourcc:
            v4l2_fourcc = fourcc_to_str(fmt.v4l2_fourcc)
        else:
            v4l2_fourcc = '    '

        planes = [f'bytespergroup:{p.bytespergroup} vsub:{p.verticalsubsampling}' for p in fmt.planes]

        print(f'{fmt.name:15} {drm_fourcc:4} {v4l2_fourcc:4} pixelspergroup:{fmt.pixelspergroup} {planes}')


if __name__ == '__main__':
    main()
