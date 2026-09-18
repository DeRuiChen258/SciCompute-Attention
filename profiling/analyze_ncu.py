#!/usr/bin/env python3
"""Summarise an ncu CSV export (raw page) into a Markdown table.

Usage: python profiling/analyze_ncu.py profiling/reports/flash_*.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

KEY_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "dram__bytes.sum",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="+")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    rows = []
    for path in args.csv_path:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                kernel = record.get("Kernel Name") or record.get("Function Name") or "unknown"
                metric = record.get("Metric Name", "")
                value = record.get("Metric Value", "")
                if metric:
                    rows.append((kernel, metric, value))

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for kernel, metric, value in rows:
        if metric in KEY_METRICS:
            grouped[kernel][metric] = value

    if not grouped:
        print("[analyze_ncu] no matching metrics found; check the CSV layout / ncu version")
        return 1

    lines = ["| kernel | " + " | ".join(KEY_METRICS) + " |",
             "| --- | " + " | ".join("---" for _ in KEY_METRICS) + " |"]
    for kernel, metrics in sorted(grouped.items()):
        cells = [metrics.get(name, "-") for name in KEY_METRICS]
        lines.append(f"| {kernel} | " + " | ".join(cells) + " |")
    report = "\n".join(lines)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report + "\n", encoding="utf-8")
        print(f"[analyze_ncu] markdown written: {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

