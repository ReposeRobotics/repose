#!/usr/bin/env python3
"""Tests for bench_to_json.parse_divan. Run: python3 -m unittest discover -s .github/scripts"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_to_json import parse_divan, parse_system_info

SAMPLE = """\
Timer precision: 41 ns
kinematics                 fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ forward_kinematics      180.5 ns      │ 244.3 ns      │ 181.9 ns      │ 184.1 ns      │ 100     │ 3200
╰─ inverse_kinematics_dls  8.291 µs      │ 11.95 µs      │ 8.374 µs      │ 8.422 µs      │ 100     │ 100

planning             fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ plan                            │               │               │               │         │
   ├─ AStar          7.068 ms      │ 9.57 ms       │ 7.323 ms      │ 7.389 ms      │ 100     │ 100
   ╰─ Direct         3.79 µs       │ 5.416 µs      │ 3.874 µs      │ 4.037 µs      │ 100     │ 100
"""


class ParseDivanTest(unittest.TestCase):
    def test_rows_groups_and_units(self):
        rows = parse_divan(SAMPLE)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["group"], "kinematics")
        self.assertEqual(rows[0]["name"], "forward_kinematics")
        self.assertEqual(rows[0]["median_ns"], 181.9)
        self.assertEqual(rows[1]["median_ns"], 8374.0)
        self.assertEqual(rows[2]["group"], "planning")
        self.assertEqual(rows[2]["name"], "plan / AStar")
        self.assertEqual(rows[2]["median_ns"], 7_323_000.0)
        self.assertEqual(rows[3]["samples"], 100)

    def test_system_info(self):
        info = parse_system_info("label: linux-x64\ncpu: Some CPU\n")
        self.assertEqual(info["cpu"], "Some CPU")


if __name__ == "__main__":
    unittest.main()
