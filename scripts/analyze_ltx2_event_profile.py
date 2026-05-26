#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


PHASE_RE = re.compile(r"^ltx2_phase::(stage[12])::step_(\d+)::(.+)$")


def adjusted_ms(row: dict[str, object], *, divide_repeated: bool) -> float:
    value = float(row["total_ms"])
    if divide_repeated and int(row.get("count", 1)) >= 2:
        value *= 0.5
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--include-step0", action="store_true")
    parser.add_argument("--no-divide-repeated", action="store_true")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    data = json.loads(args.profile.read_text())
    stats = data["stats"]
    roots: collections.defaultdict[tuple[str, str], float] = collections.defaultdict(float)
    labels: collections.defaultdict[tuple[str, str], float] = collections.defaultdict(float)
    per_step: collections.defaultdict[tuple[str, int], float] = collections.defaultdict(float)
    block_roots: collections.defaultdict[tuple[str, str], float] = collections.defaultdict(float)

    for row in stats:
        match = PHASE_RE.match(str(row["name"]))
        if match is None:
            continue
        stage = match.group(1)
        step = int(match.group(2))
        rest = match.group(3)
        if step == 0 and not args.include_step0:
            continue

        value = adjusted_ms(row, divide_repeated=not args.no_divide_repeated)
        root = rest.split("::", 1)[0]
        roots[(stage, root)] += value
        labels[(stage, rest)] += value
        if root == "ltx2_dit_block":
            per_step[(stage, step)] += value

        parts = rest.split("::")
        if len(parts) >= 2 and parts[0] == "ltx2_dit_block":
            block_roots[(stage, parts[1])] += value

    print("Root totals")
    for (stage, root), value in sorted(roots.items(), key=lambda item: item[1], reverse=True)[
        : args.top
    ]:
        print(f"{value / 1000:9.3f}s  {stage:6s}  {root}")

    print("\nTop labels")
    for (stage, label), value in sorted(
        labels.items(), key=lambda item: item[1], reverse=True
    )[: args.top]:
        print(f"{value / 1000:9.3f}s  {stage:6s}  {label}")

    print("\nBlock totals")
    for (stage, block), value in sorted(
        block_roots.items(), key=lambda item: item[1], reverse=True
    )[: args.top]:
        print(f"{value / 1000:9.3f}s  {stage:6s}  block_{block}")

    if per_step:
        print("\nStep block totals")
        for (stage, step), value in sorted(per_step.items())[: args.top]:
            print(f"{value / 1000:9.3f}s  {stage:6s}  step_{step}")


if __name__ == "__main__":
    main()
