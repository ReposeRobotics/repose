#!/usr/bin/env python3
"""Convert Divan text output into the benchmark JSON consumed by the website.

Usage:
    bench_to_json.py <divan.txt> <system-info.txt> <out.json>
        --label linux-x64 --commit <sha> --version 0.8.2 --generated-at <iso>

Every timing is normalized to nanoseconds. Lines without a timing row are
tree/group headers; timed leaves inherit the current header path.
"""

import argparse
import json
import re
import sys

UNITS_NS = {"ns": 1.0, "µs": 1_000.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}

# e.g. "forward_kinematics  180.5 ns │ 244.3 ns │ 181.9 ns │ 184.1 ns │ 100 │ 3200"
TIMED = re.compile(
    r"^(?P<name>\S(?:.*?\S)?)\s+"
    r"(?P<fastest>[\d.]+)\s*(?P<fastest_unit>ns|µs|us|ms|s)\s*│\s*"
    r"(?P<slowest>[\d.]+)\s*(?P<slowest_unit>ns|µs|us|ms|s)\s*│\s*"
    r"(?P<median>[\d.]+)\s*(?P<median_unit>ns|µs|us|ms|s)\s*│\s*"
    r"(?P<mean>[\d.]+)\s*(?P<mean_unit>ns|µs|us|ms|s)\s*│\s*"
    r"(?P<samples>\d+)\s*│\s*(?P<iters>\d+)\s*$"
)
TREE_CHARS = " ├╰│─"


def to_ns(value, unit):
    return round(float(value) * UNITS_NS[unit], 3)


def parse_divan(text):
    results = []
    path = []  # (depth, header) stack for untimed group lines
    for raw in text.splitlines():
        if "fastest" in raw and "slowest" in raw:
            # Column header; its first word is the bench binary title.
            title = raw.split()[0].lstrip(TREE_CHARS)
            path = [(0, title)]
            continue
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip(TREE_CHARS)
        depth = (len(line) - len(stripped)) // 3
        match = TIMED.match(stripped)
        if match:
            group = path[0][1] if path else ""
            name = " / ".join(
                [h for d, h in path[1:] if d < depth] + [match["name"]]
            )
            results.append(
                {
                    "group": group,
                    "name": name,
                    "fastest_ns": to_ns(match["fastest"], match["fastest_unit"]),
                    "slowest_ns": to_ns(match["slowest"], match["slowest_unit"]),
                    "median_ns": to_ns(match["median"], match["median_unit"]),
                    "mean_ns": to_ns(match["mean"], match["mean_unit"]),
                    "samples": int(match["samples"]),
                    "iters": int(match["iters"]),
                }
            )
        elif stripped and not stripped.startswith(("Timer precision", "Running ")):
            # Group header such as `plan` (may carry empty timing cells).
            header = stripped.split("│")[0].strip()
            if header:
                path = [(d, h) for d, h in path if d < depth]
                path.append((depth, header))
    return results


def parse_system_info(text):
    info = {}
    for line in text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            info[key.strip()] = value.strip()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("divan")
    parser.add_argument("system_info")
    parser.add_argument("out")
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()

    with open(args.divan, encoding="utf-8") as handle:
        results = parse_divan(handle.read())
    if not results:
        print("error: no benchmark rows parsed", file=sys.stderr)
        return 1
    with open(args.system_info, encoding="utf-8") as handle:
        info = parse_system_info(handle.read())

    payload = {
        "schema": 1,
        "label": args.label,
        "runner": info.get("os", ""),
        "os": info.get("uname", ""),
        "arch": info.get("arch", ""),
        "cpu": info.get("cpu", ""),
        "rustc": info.get("rustc", ""),
        "commit": args.commit,
        "version": args.version,
        "generated_at": args.generated_at,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"{args.label}: {len(results)} benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
