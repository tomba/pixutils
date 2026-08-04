#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys

SHARED_KEYS = ('width', 'height', 'padding', 'backends', 'options')


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if 'meta' not in data or 'runs' not in data:
        sys.exit(f'{path}: not a perf-test result file (missing meta/runs)')
    return data


def validate_shared_config(files: list[tuple[str, dict]]) -> None:
    base_path, base_data = files[0]
    base_meta = base_data['meta']
    mismatches: list[str] = []
    for path, data in files[1:]:
        meta = data['meta']
        for key in SHARED_KEYS:
            if meta.get(key) != base_meta.get(key):
                mismatches.append(
                    f'  {key}: {base_path}={base_meta.get(key)!r} vs {path}={meta.get(key)!r}'
                )
    if mismatches:
        sys.exit('Shared config differs between files:\n' + '\n'.join(mismatches))


def mean_peak(run: dict) -> tuple[float, float]:
    return run['iters_per_s'], 1.0 / run['min_iter_s']


def commit_label(meta: dict) -> str:
    commit = meta.get('git_commit') or '-'
    if meta.get('git_dirty'):
        commit += '*'
    return commit


def format_options(options: dict) -> str:
    if not options:
        return '{}'
    return '{' + ', '.join(f'{k}:{v}' for k, v in options.items()) + '}'


def print_table(rows: list[list[str]], aligns: list[str]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r_idx, row in enumerate(rows):
        parts = []
        for i, cell in enumerate(row):
            if aligns[i] == 'l':
                parts.append(cell.ljust(widths[i]))
            else:
                parts.append(cell.rjust(widths[i]))
        print('  '.join(parts).rstrip())
        if r_idx == 0:
            print('  '.join('-' * w for w in widths))


def format_order_union(files: list[tuple[str, dict]]) -> list[str]:
    """Baseline format order, then any formats that appear only in later files."""
    baseline_formats = [r['format'] for r in files[0][1]['runs']]
    seen = set(baseline_formats)
    extras: list[str] = []
    for _, data in files[1:]:
        for r in data['runs']:
            if r['format'] not in seen:
                extras.append(r['format'])
                seen.add(r['format'])
    return baseline_formats + extras


def print_default(files: list[tuple[str, dict]], sort: str) -> None:
    file_rows: list[list[str]] = [['File', 'Commit', 'When']]
    for path, data in files:
        m = data['meta']
        ts = m.get('timestamp', '').replace('T', ' ').rsplit('+', 1)[0].rsplit('-', 0)[0][:16]
        file_rows.append([os.path.basename(path), commit_label(m), ts])
    print_table(file_rows, ['l', 'l', 'l'])
    print()

    by_file: list[dict[str, tuple[float, float]]] = [
        {r['format']: mean_peak(r) for r in data['runs']} for _, data in files
    ]
    format_order = format_order_union(files)

    if sort == 'baseline':
        format_order.sort(key=lambda f: -(by_file[0].get(f, (0.0, 0.0))[1]))
    elif sort == 'delta' and len(files) >= 2:

        def delta_key(f: str) -> float:
            b = by_file[0].get(f)
            c = by_file[1].get(f)
            if b is None or c is None:
                return float('inf')
            return (c[1] - b[1]) / b[1]

        format_order.sort(key=delta_key)

    base_name = os.path.basename(files[0][0])
    header = ['Format', f'{base_name} mean', f'{base_name} peak']
    aligns = ['l', 'r', 'r']
    for path, _ in files[1:]:
        name = os.path.basename(path)
        header.extend([f'{name} mean', f'{name} peak', 'mean', 'peak'])
        aligns.extend(['r', 'r', 'r', 'r'])

    rows: list[list[str]] = [header]
    for fmt in format_order:
        base = by_file[0].get(fmt)
        if base is None:
            row = [fmt, '—', '—']
        else:
            row = [fmt, f'{base[0]:.1f}/s', f'{base[1]:.1f}/s']
        for d in by_file[1:]:
            v = d.get(fmt)
            if v is None:
                row.extend(['—', '—', '—', '—'])
            elif base is None:
                row.extend([f'{v[0]:.1f}/s', f'{v[1]:.1f}/s', '—', '—'])
            else:
                pct_mean = (v[0] - base[0]) / base[0] * 100.0
                pct_peak = (v[1] - base[1]) / base[1] * 100.0
                row.extend(
                    [f'{v[0]:.1f}/s', f'{v[1]:.1f}/s', f'{pct_mean:+.1f}%', f'{pct_peak:+.1f}%']
                )
        rows.append(row)

    print_table(rows, aligns)


def print_timeline_for_format(files: list[tuple[str, dict]], fmt: str) -> None:
    print(f'Format: {fmt}')
    vals: list[tuple[float, float] | None] = [
        next(
            (mean_peak(r) for r in data['runs'] if r['format'] == fmt),
            None,
        )
        for _, data in files
    ]
    base = vals[0]
    prev = base

    rows: list[list[str]] = [
        [
            'Commit',
            'mean',
            'peak',
            'base mean',
            'base peak',
            'prev mean',
            'prev peak',
            'Subject',
        ]
    ]
    for i, ((_, data), v) in enumerate(zip(files, vals)):
        row = [commit_label(data['meta'])]
        if v is None:
            row.extend(['—', '—'])
        else:
            row.extend([f'{v[0]:.1f}/s', f'{v[1]:.1f}/s'])

        if i == 0 or v is None or base is None:
            row.extend(['—', '—'])
        else:
            row.extend(
                [
                    f'{(v[0] - base[0]) / base[0] * 100:+.1f}%',
                    f'{(v[1] - base[1]) / base[1] * 100:+.1f}%',
                ]
            )

        if i == 0 or v is None or prev is None:
            row.extend(['—', '—'])
        else:
            row.extend(
                [
                    f'{(v[0] - prev[0]) / prev[0] * 100:+.1f}%',
                    f'{(v[1] - prev[1]) / prev[1] * 100:+.1f}%',
                ]
            )

        row.append(data['meta'].get('git_subject') or '')
        rows.append(row)
        if v is not None:
            prev = v

    print_table(rows, ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l'])


def print_timeline(files: list[tuple[str, dict]]) -> None:
    format_order = format_order_union(files)
    for i, fmt in enumerate(format_order):
        if i > 0:
            print()
        print_timeline_for_format(files, fmt)


def main():
    parser = argparse.ArgumentParser(description='Compare conv-perf-test.py JSON results.')
    parser.add_argument('files', nargs='+', help='Result files (first is baseline)')
    parser.add_argument(
        '--sort',
        choices=['format', 'delta', 'baseline'],
        default='format',
        help="Row sort order (default: 'format' = baseline order)",
    )
    parser.add_argument(
        '--timeline',
        action='store_true',
        help='Transposed layout: commits as rows, formats as columns, with Δbase and Δprev',
    )
    args = parser.parse_args()

    if len(args.files) < 2:
        sys.exit('Need at least 2 files to compare')

    files = [(p, load(p)) for p in args.files]
    validate_shared_config(files)

    base_meta = files[0][1]['meta']
    print(
        f'Config: {base_meta["width"]}x{base_meta["height"]}  '
        f'padding={base_meta["padding"]}  '
        f'backends={",".join(base_meta["backends"])}  '
        f'options={format_options(base_meta["options"])}'
    )
    print()

    if args.timeline:
        print_timeline(files)
    else:
        print_default(files, args.sort)


if __name__ == '__main__':
    main()
