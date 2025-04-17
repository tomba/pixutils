#!/usr/bin/env python3
"""
Color Conversion Matrix Generator

This script generates conversion matrices for color space conversions
(YCbCr <-> RGB) according to various ITU standards, in both full and limited range formats.

Main references:
- ITU-R BT.601: https://www.itu.int/rec/R-REC-BT.601
- ITU-R BT.709: https://www.itu.int/rec/R-REC-BT.709
- ITU-R BT.2020: https://www.itu.int/rec/R-REC-BT.2020

Usage:
  python color_conversion.py --direction=yuv2rgb --standard=bt601 --range=limited --output=text
"""

import argparse


def generate_conversion_matrix(
    direction='yuv2rgb', standard='bt601', range_type='full', input_order='YCbCr', transpose=False
):
    """
    Generate conversion matrix and offsets for color space conversion.

    Args:
        direction: 'yuv2rgb' or 'rgb2yuv'
        standard: 'bt601', 'bt709', or 'bt2020'
        range_type: 'full' or 'limited'
        input_order: 'YCbCr' or 'YCrCb' to indicate order of YUV channels
        transpose: Whether to transpose the final coefficient matrix

    Returns:
        Dictionary with matrix, pre_offsets, and post_offsets
    """
    # Define Kr and Kb values according to standards
    kr_kb = {
        'bt601': (0.299, 0.114),  # ITU-R BT.601, TABLE 1 "Normalized signal values"
        'bt709': (0.2126, 0.0722),  # ITU-R BT.709, 3 "Signal format"
        'bt2020': (0.2627, 0.0593),  # ITU-R BT.2020, TABLE 4, "Signal format"
    }

    # Get Kr and Kb for the requested standard
    Kr, Kb = kr_kb[standard]
    Kg = 1 - Kr - Kb

    if direction == 'yuv2rgb':
        # Calculate YUV to RGB matrix coefficients
        Ry = 1.0
        Rv = 2.0 * (1 - Kr)
        Gy = 1.0
        Gv = -2.0 * (1 - Kr) * Kr / Kg
        Gu = -2.0 * (1 - Kb) * Kb / Kg
        By = 1.0
        Bu = 2.0 * (1 - Kb)

        # Scale for limited range if necessary
        y_scale = 255.0 / 219.0 if range_type == 'limited' else 1.0
        c_scale = 255.0 / 224.0 if range_type == 'limited' else 1.0

        # Apply scaling to all coefficients
        Ry *= y_scale
        Rv *= c_scale
        Gy *= y_scale
        Gv *= c_scale
        Gu *= c_scale
        By *= y_scale
        Bu *= c_scale

        # Set up offsets
        y_offset = -16.0 / 255.0 if range_type == 'limited' else 0.0
        c_offset = -0.5  # This is always -0.5 to center Cb/Cr around 0

        # Create coefficient matrix based on input order
        if input_order == 'ycrcb':
            # YCrCb order (Cr before Cb)
            matrix = [
                [Ry, Rv, 0.0],  # R = Y + Rv*Cr
                [Gy, Gv, Gu],  # G = Y + Gv*Cr + Gu*Cb
                [By, 0.0, Bu],  # B = Y + Bu*Cb
            ]
            in_channels = ['Y', 'Cr', 'Cb']
        else:
            # YCbCr order (standard)
            matrix = [
                [Ry, 0.0, Rv],  # R = Y + Rv*Cr
                [Gy, Gu, Gv],  # G = Y + Gu*Cb + Gv*Cr
                [By, Bu, 0.0],  # B = Y + Bu*Cb
            ]
            in_channels = ['Y', 'Cb', 'Cr']

        out_channels = ['R', 'G', 'B']
        pre_offsets = [y_offset, c_offset, c_offset]
        post_offsets = [0.0, 0.0, 0.0]

    else:  # RGB to YUV conversion
        # Scale factors for limited range
        y_scale = 219.0 / 255.0 if range_type.lower() == 'limited' else 1.0
        c_scale = 224.0 / 255.0 if range_type.lower() == 'limited' else 1.0

        # RGB to YUV matrix - using standard formulas
        # Y = Kr*R + Kg*G + Kb*B
        # Cb = (B-Y)/(2*(1-Kb))
        # Cr = (R-Y)/(2*(1-Kr))

        matrix = [
            [Kr, Kg, Kb],                                  # Y = Kr*R + Kg*G + Kb*B
            [-Kr/(2*(1-Kb)), -Kg/(2*(1-Kb)), 0.5],        # Cb = (B-Y)/(2*(1-Kb))
            [0.5, -Kg/(2*(1-Kr)), -Kb/(2*(1-Kr))]         # Cr = (R-Y)/(2*(1-Kr))
        ]

        # Apply scaling for limited range
        for i in range(3):
            scale = y_scale if i == 0 else c_scale
            for j in range(3):
                matrix[i][j] *= scale

        # Setup offsets
        pre_offsets = [0.0, 0.0, 0.0]
        y_offset = 16.0/255.0 if range_type.lower() == 'limited' else 0.0
        c_offset = 128.0/255.0  # For both Cb and Cr
        post_offsets = [y_offset, c_offset, c_offset]

        in_channels = ['R', 'G', 'B']

        # Adjust output channel order if needed
        if input_order.lower() == 'ycrcb':
            # Swap Cb and Cr in the matrix and offsets
            matrix[1], matrix[2] = matrix[2], matrix[1]
            post_offsets[1], post_offsets[2] = post_offsets[2], post_offsets[1]
            out_channels = ['Y', 'Cr', 'Cb']
        else:
            out_channels = ['Y', 'Cb', 'Cr']

    # Transpose matrix if requested
    if transpose:
        matrix = [[matrix[j][i] for j in range(3)] for i in range(3)]

    return {
        'direction': direction,
        'range': range_type,
        'matrix': matrix,
        'pre_offsets': pre_offsets,
        'post_offsets': post_offsets,
        'in_channels': in_channels,
        'out_channels': out_channels,
    }


def generate_y_only_matrix(range_type='full', direction='yuv2rgb', transpose=False):
    """
    Generate matrix for Y-only (grayscale) conversion.

    Args:
        range_type: 'full' or 'limited'
        direction: 'yuv2rgb' or 'rgb2yuv'
        transpose: Whether to transpose the final matrix

    Returns:
        Dictionary with matrix, pre_offsets, and post_offsets
    """
    if direction == 'yuv2rgb':
        # For Y-only YUV to RGB, we just copy Y to all RGB channels
        y_scale = 255.0 / 219.0 if range_type == 'limited' else 1.0
        y_offset = -16.0 / 255.0 if range_type == 'limited' else 0.0

        matrix = [
            [0.0, 0.0, y_scale],  # R = Y
            [0.0, 0.0, y_scale],  # G = Y
            [0.0, 0.0, y_scale],  # B = Y
        ]

        pre_offsets = [0.0, 0.0, y_offset]
        post_offsets = [0.0, 0.0, 0.0]
        in_channels = ['ignored', 'ignored', 'Y']
        out_channels = ['R', 'G', 'B']
    else:
        # For Y-only RGB to YUV, we take average of RGB (or use standard luminance)
        # Using standard BT.709 luminance coefficients
        y_scale = 219.0 / 255.0 if range_type == 'limited' else 1.0

        matrix = [
            [0.2126, 0.7152, 0.0722],  # Y = 0.2126*R + 0.7152*G + 0.0722*B (BT.709)
            [0.0, 0.0, 0.0],  # Cb = 0 (no color)
            [0.0, 0.0, 0.0],  # Cr = 0 (no color)
        ]

        # Apply scaling for limited range
        for j in range(3):
            matrix[0][j] *= y_scale

        pre_offsets = [0.0, 0.0, 0.0]
        y_offset = 16.0 / 255.0 if range_type == 'limited' else 0.0
        post_offsets = [y_offset, 128.0 / 255.0, 128.0 / 255.0]  # Cb, Cr at 128 (gray)

        in_channels = ['R', 'G', 'B']
        out_channels = ['Y', 'Cb', 'Cr']

    # Transpose matrix if requested
    if transpose:
        matrix = [[matrix[j][i] for j in range(3)] for i in range(3)]
        # When transposing, the channel order is swapped for interpretation
        in_channels, out_channels = out_channels, in_channels

    return {
        'matrix': matrix,
        'pre_offsets': pre_offsets,
        'post_offsets': post_offsets,
        'in_channels': in_channels,
        'out_channels': out_channels,
    }


def format_output_xilinx(conversion_data):
    """Format conversion data for Xilinx driver format."""
    coeff_format = '\t\t.coeffs = {\n'
    for row in conversion_data['matrix']:
        coeff_format += f'\t\t\tC({row[0]:9.6f}), C({row[1]:9.6f}), C({row[2]:9.6f}),\n'
    coeff_format += '\t\t},\n'

    pre_format = '\t\t.pre_offsets = {\n\t\t\t'
    pre_format += ', '.join(f'O({v:7.4f})' for v in conversion_data['pre_offsets'])
    pre_format += ',\n\t\t},\n'

    post_format = '\t\t.post_offsets = {\n\t\t\t'
    post_format += ', '.join(f'O({v:7.4f})' for v in conversion_data['post_offsets'])
    post_format += ',\n\t\t},\n'

    return coeff_format + pre_format + post_format


def format_output_text(conversion_data):
    """Format conversion data as human-readable text."""
    result = 'Matrix:\n'
    for i, row in enumerate(conversion_data['matrix']):
        result += f"  {conversion_data['out_channels'][i]} = "
        terms = []
        for j, val in enumerate(row):
            if val != 0:
                terms.append(f"{val:.6f}*{conversion_data['in_channels'][j]}")
        result += ' + '.join(terms) + '\n'

    result += '\nPre-offsets (applied to input):\n'
    for i, offset in enumerate(conversion_data['pre_offsets']):
        if offset != 0:
            result += f"  {conversion_data['in_channels'][i]}: {offset:.6f}\n"

    result += '\nPost-offsets (applied to output):\n'
    for i, offset in enumerate(conversion_data['post_offsets']):
        if offset != 0:
            result += f"  {conversion_data['out_channels'][i]}: {offset:.6f}\n"

    return result


def format_output_test(conversion_data):
    """Format conversion data as a set of test color conversions using 8-bit integer values."""
    result = ''

    # Determine direction based on in/out channels
    if conversion_data['direction'] == 'rgb2yuv':
        # Define test colors in RGB (8-bit integers 0-255)
        input_colors = {
            'Black':  [  0,   0,   0],
            'White':  [255, 255, 255],
            'Red':    [255,   0,   0],
            'Green':  [  0, 255,   0],
            'Blue':   [  0,   0, 255],
            'Gray50': [128, 128, 128]
        }
    else:
        # Define test colors in YCbCr
        if conversion_data['range'] == 'full':
            input_colors = {
                'Black':  [  0, 128, 128],
                'White':  [255, 128, 128],
                'Red':    [ 76,  85, 255],
                'Green':  [150,  44,  21],
                'Blue':   [ 29, 255, 107],
                'Gray50': [128, 128, 128],
            }
        else:
            input_colors = {
                'Black':  [ 16, 128, 128],
                'White':  [235, 128, 128],
                'Red':    [ 81,  90, 240],
                'Green':  [145,  54,  34],
                'Blue':   [ 41, 240, 110],
                'Gray50': [126, 128, 128],
            }

    # Get matrix and offsets
    matrix = conversion_data['matrix']
    pre_offsets = conversion_data['pre_offsets']
    post_offsets = conversion_data['post_offsets']

    result += '\n{:<10} {:<25} {:<25}\n'.format('Color', 'Input (8-bit)', 'Output (8-bit)')
    result += '-' * 60 + '\n'

    # For each test color
    for color_name, color_values_int in input_colors.items():
        # Convert 8-bit integers to normalized values (0-1)
        color_values = [v / 255.0 for v in color_values_int]

        # Apply pre-offsets
        adjusted_input = [color_values[i] + pre_offsets[i] for i in range(3)]

        # Apply matrix multiplication
        output = [0, 0, 0]
        for i in range(3):
            for j in range(3):
                output[i] += matrix[i][j] * adjusted_input[j]

        # Apply post-offsets
        output = [output[i] + post_offsets[i] for i in range(3)]

        # Convert normalized output back to 8-bit integers
        output_int = [max(0, min(255, round(v * 255))) for v in output]

        # Format input and output nicely (as integers)
        input_str = '[{:3d}, {:3d}, {:3d}]'.format(*color_values_int)
        output_str = '[{:3d}, {:3d}, {:3d}]'.format(*output_int)

        # Add to result
        result += '{:<10} {:<25} {:<25}\n'.format(color_name, input_str, output_str)

    return result


# Dictionary mapping format types to their formatter functions
FORMATTERS = {
    'xilinx': format_output_xilinx,
    'text': format_output_text,
    'test': format_output_test,
}


def format_output(conversion_data, format_type):
    """
    Format conversion data in different output formats.

    Args:
        conversion_data: Dictionary with matrix, pre_offsets, post_offsets
        format_type: 'xilinx', 'text', etc.

    Returns:
        String with formatted output
    """
    formatter = FORMATTERS.get(format_type)
    if formatter:
        return formatter(conversion_data)
    else:
        # Default to text format if unknown format is requested
        return format_output_text(conversion_data)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate color space conversion matrices.')

    parser.add_argument(
        '--direction',
        choices=['yuv2rgb', 'rgb2yuv'],
        default='yuv2rgb',
        help='Direction of color conversion (default: yuv2rgb)',
    )

    parser.add_argument(
        '--standard',
        choices=['bt601', 'bt709', 'bt2020', 'all'],
        default='all',
        help='Color standard to use (default: all)',
    )

    parser.add_argument(
        '--range',
        choices=['full', 'limited', 'all'],
        default='all',
        help='Color range to use (default: all)',
    )

    parser.add_argument(
        '--format',
        choices=['xilinx', 'text', 'test'],
        default='text',
        help='Output format (default: text)',
    )

    parser.add_argument(
        '--order',
        choices=['ycbcr', 'ycrcb'],
        default='ycbcr',
        help='Order of YUV components (default: ycbcr)',
    )

    parser.add_argument('--transpose', action='store_true', help='Transpose the conversion matrix')

    parser.add_argument(
        '--y-only', action='store_true', help='Generate Y-only (grayscale) conversion matrix'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Determine which standards and ranges to process
    standards = ['bt601', 'bt709', 'bt2020'] if args.standard == 'all' else [args.standard]
    ranges = ['full', 'limited'] if args.range == 'all' else [args.range]

    for std in standards:
        for rng in ranges:
            label = f'{std.upper()} {rng.capitalize()} Range'
            direction_label = 'YUV to RGB' if args.direction == 'yuv2rgb' else 'RGB to YUV'
            order_label = 'YCrCb' if args.order == 'ycrcb' else 'YCbCr'
            transpose_label = ' (Transposed)' if args.transpose else ''

            print(f'\n=== {label} - {direction_label} - {order_label}{transpose_label} ===')

            if args.y_only:
                data = generate_y_only_matrix(rng, args.direction, args.transpose)
                print(format_output(data, args.format))
            else:
                data = generate_conversion_matrix(
                    args.direction, std, rng, args.order, args.transpose
                )
                print(format_output(data, args.format))


if __name__ == '__main__':
    main()
