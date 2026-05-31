#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_tensors(value: Any, prefix: str = ''):
    if isinstance(value, dict):
        if value.get('kind') == 'tensor':
            yield prefix or '<tensor>', value
            return
        items = value.get('items')
        if isinstance(items, dict):
            for key, child in items.items():
                yield from iter_tensors(child, f'{prefix}.{key}' if prefix else str(key))
            return
        if isinstance(items, list):
            for idx, child in enumerate(items):
                yield from iter_tensors(child, f'{prefix}[{idx}]')
            return
        for key, child in value.items():
            if key in {'kind', 'len'}:
                continue
            yield from iter_tensors(child, f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from iter_tensors(child, f'{prefix}[{idx}]')


def fmt_tensor(t: dict[str, Any]) -> str:
    fields = [
        f"shape={t.get('shape')}",
        f"dtype={t.get('dtype')}",
        f"mean={t.get('mean')}",
        f"std={t.get('std')}",
        f"min={t.get('min')}",
        f"max={t.get('max')}",
        f"l2={t.get('l2')}",
        f"sha={str(t.get('sha256', ''))[:16]}",
    ]
    return ', '.join(fields)


def compare_tensor_trees(a: Any, b: Any):
    a_tensors = dict(iter_tensors(a))
    b_tensors = dict(iter_tensors(b))
    all_paths = sorted(set(a_tensors) | set(b_tensors))
    for path in all_paths:
        at = a_tensors.get(path)
        bt = b_tensors.get(path)
        if at is None or bt is None:
            return path, at, bt, 'missing_tensor'
        if at.get('shape') != bt.get('shape') or at.get('dtype') != bt.get('dtype'):
            return path, at, bt, 'shape_or_dtype'
        if at.get('sha256') != bt.get('sha256'):
            return path, at, bt, 'sha256'
    return None, None, None, ''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()

    a_rows = load_jsonl(Path(args.a))
    b_rows = load_jsonl(Path(args.b))
    print(f'a_rows={len(a_rows)} b_rows={len(b_rows)}')
    limit = min(len(a_rows), len(b_rows))
    mismatches = 0
    for idx in range(limit):
        a = a_rows[idx]
        b = b_rows[idx]
        path, at, bt, reason = compare_tensor_trees(a.get('output', a), b.get('output', b))
        name_same = a.get('name') == b.get('name')
        class_same = a.get('module_class') == b.get('module_class')
        if reason or not name_same or not class_same:
            mismatches += 1
            print(f'\n[event {idx}] reason={reason or "metadata"}')
            print(f"  A name={a.get('name')} class={a.get('module_class')} forward={a.get('forward_index')}")
            print(f"  B name={b.get('name')} class={b.get('module_class')} forward={b.get('forward_index')}")
            if path is not None:
                print(f'  tensor_path={path}')
                print(f'  A {fmt_tensor(at) if at else None}')
                print(f'  B {fmt_tensor(bt) if bt else None}')
            if mismatches >= args.limit:
                break
    if len(a_rows) != len(b_rows):
        print(f'row_count_diff a={len(a_rows)} b={len(b_rows)}')
    if mismatches == 0 and len(a_rows) == len(b_rows):
        print('all compared events match')


if __name__ == '__main__':
    main()
