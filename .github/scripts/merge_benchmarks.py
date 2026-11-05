#!/usr/bin/env python3
"""Merge per-system benchmark JSON files into one benchmarks.json.

Usage: merge_benchmarks.py <out.json> <system.json>...
"""

import json
import sys


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    systems = []
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as handle:
            systems.append(json.load(handle))
    systems.sort(key=lambda system: system["label"])
    first = systems[0]
    payload = {
        "schema": 1,
        "version": first["version"],
        "commit": first["commit"],
        "generated_at": max(system["generated_at"] for system in systems),
        "systems": systems,
    }
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"merged {len(systems)} systems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
