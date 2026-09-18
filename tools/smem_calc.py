#!/usr/bin/env python3
"""Shared-memory / register budget calculator for the tiled and flash tile tables.

This mirrors the C++ arithmetic in `include/scicompute_attention/detail/tile_config.hpp`
(`SmemBytesFlash`) and `src/backends/flash/flash_tile_config.hpp`; `tests/unit/test_tile_smem_calc.cpp`
pins the C++ side, and this script lets a tile sweep be planned without a GPU.

Usage:
  python tools/smem_calc.py --dtype fp16 --head-dims 32,64,96,128,160,192,256
  python tools/smem_calc.py --dtype fp16 --tile 64 64 4 2 --head-dim 128
"""

from __future__ import annotations

import argparse
import json
import sys

SMEM_LIMIT_BYTES = 101376  # measured cudaDevAttrMaxSharedMemoryPerBlockOptin (docs/env_report.md)
ROW_PAD_ELEMS = 8          # 16 B row padding (flash_smem_layout / arch_features.cuh)

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}

# Seed table from src/backends/flash/flash_tile_config.hpp (must stay identical).
FLASH_TILES = {
    32: (64, 64, 4, 2, 1),
    64: (64, 64, 4, 2, 1),
    96: (64, 64, 4, 2, 1),
    128: (64, 64, 4, 2, 1),
    160: (64, 32, 8, 2, 2),
    192: (64, 32, 8, 2, 2),
    256: (32, 32, 4, 2, 2),
}


def flash_smem_bytes(head_dim: int, elem_bytes: int, block_m: int, block_n: int,
                     stages: int) -> int:
    row_stride = head_dim + ROW_PAD_ELEMS
    q = block_m * row_stride * elem_bytes
    kv = block_n * row_stride * elem_bytes * stages
    return q + 2 * kv


def register_budget(head_dim: int, block_n: int, splits_d: int) -> dict:
    """Rough per-thread register estimate for the flash kernel (documented approximation)."""
    dtiles_per_warp = (head_dim // 8) // splits_d
    o_acc = dtiles_per_warp * 4
    q_frag = (head_dim // 16) * 4
    s_frag = (block_n // 8) * 4
    return {"o_acc": o_acc, "q_frag": q_acc if (q_acc := q_frag) else 0, "s_frag": s_frag,
            "estimate_total": o_acc + q_frag + s_frag + 24}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPE_BYTES))
    parser.add_argument("--head-dims", default="32,64,96,128,160,192,256")
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--tile", nargs=4, type=int, metavar=("BM", "BN", "WARPS", "STAGES"))
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    elem = DTYPE_BYTES[args.dtype]
    rows = []
    if args.tile:
        if not args.head_dim:
            print("--tile requires --head-dim", file=sys.stderr)
            return 2
        block_m, block_n, warps, stages = args.tile
        smem = flash_smem_bytes(args.head_dim, elem, block_m, block_n, stages)
        rows.append({"head_dim": args.head_dim, "block_m": block_m, "block_n": block_n,
                     "warps": warps, "stages": stages, "splits_d": 1, "smem_bytes": smem,
                     "fits": smem <= SMEM_LIMIT_BYTES,
                     "regs": register_budget(args.head_dim, block_n, 1)})
    else:
        for head_dim in [int(x) for x in args.head_dims.split(",")]:
            tile = FLASH_TILES.get(head_dim)
            if tile is None:
                continue
            block_m, block_n, warps, stages, splits_d = tile
            smem = flash_smem_bytes(head_dim, elem, block_m, block_n, stages)
            rows.append({"head_dim": head_dim, "block_m": block_m, "block_n": block_n,
                         "warps": warps, "stages": stages, "splits_d": splits_d,
                         "smem_bytes": smem, "fits": smem <= SMEM_LIMIT_BYTES,
                         "regs": register_budget(head_dim, block_n, splits_d)})

    print(f"{'D':>5} {'BM':>4} {'BN':>4} {'warps':>5} {'stages':>6} {'splits':>6} "
          f"{'smem B':>8} {'limit':>7} {'OK':>3} {'regs~':>6}")
    for row in rows:
        print(f"{row['head_dim']:>5} {row['block_m']:>4} {row['block_n']:>4} {row['warps']:>5} "
              f"{row['stages']:>6} {row['splits_d']:>6} {row['smem_bytes']:>8} "
              f"{SMEM_LIMIT_BYTES:>7} {'yes' if row['fits'] else 'NO':>3} "
              f"{row['regs']['estimate_total']:>6}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"dtype": args.dtype, "smem_limit_bytes": SMEM_LIMIT_BYTES,
                       "row_pad_elems": ROW_PAD_ELEMS, "rows": rows}, handle, indent=2)
        print(f"[smem_calc] json written: {args.json}")
    return 0 if all(row["fits"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

