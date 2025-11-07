#!/usr/bin/env python3
"""Sweep KO_STEP / tile / micro-batch combos for fused XPU convs."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from pathlib import Path


def parse_list(val: str, cast=int) -> list:
    vals = []
    for tok in val.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(cast(tok))
    return vals


def parse_optional_list(val: str, cast=int) -> list:
    val = (val or "").strip()
    if not val:
        return [None]
    return parse_list(val, cast=cast)


def run_one(cfg: dict, env: dict, dbg_max: int) -> tuple[float, float]:
    cmd = [
        sys.executable,
        "-u",
        "src/train_xpu_overlap.py",
        "--config",
        cfg["config"],
        "--fused",
        "on",
        "--channels-last",
        "on",
        "--dbg-max-steps",
        str(dbg_max),
    ]
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env.setdefault("DBG_TIMING", "1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=merged_env)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"run failed (code={proc.returncode})\n{out}")
    f_ms = b_ms = None
    for line in out.splitlines():
        if line.startswith("[dbg] step=1"):
            parts = line.split()
            # ... fwd=xxxxms ... total=xxxms
            for part in parts:
                if part.startswith("fwd="):
                    f_ms = float(part.split("=")[1][:-2])
                if part.startswith("total="):
                    b_ms = float(part.split("=")[1][:-2])
            break
    if f_ms is None or b_ms is None:
        raise RuntimeError(f"timing not found\n{out}")
    return f_ms, b_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar10_xpu_fused.yaml")
    ap.add_argument("--ko-step", default="8,16,32")
    ap.add_argument("--tile-h", default="8,16")
    ap.add_argument("--tile-w", default="16,32")
    ap.add_argument("--micro-n", default="0,16")
    ap.add_argument("--dbg-max-steps", type=int, default=3)
    ap.add_argument("--output", default="outputs/fused_sweep.csv")
    ap.add_argument("--resume", action="store_true", help="Skip combos already logged in the output CSV")
    ap.add_argument("--ko-step-s2", default="")
    ap.add_argument("--tile-h-s2", default="")
    ap.add_argument("--tile-w-s2", default="")
    ap.add_argument("--micro-n-s2", default="")
    args = ap.parse_args()

    ko_vals = parse_list(args.ko_step)
    th_vals = parse_list(args.tile_h)
    tw_vals = parse_list(args.tile_w)
    micro_vals = parse_list(args.micro_n)
    ko_s2_vals = parse_optional_list(args.ko_step_s2)
    th_s2_vals = parse_optional_list(args.tile_h_s2)
    tw_s2_vals = parse_optional_list(args.tile_w_s2)
    micro_s2_vals = parse_optional_list(args.micro_n_s2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[int, int, int, int, int | None, int | None, int | None, int | None]] = set()
    results: list[tuple[int, int, int, int, int | None, int | None, int | None, int | None, float, float]] = []
    if out_path.exists() and args.resume:
        with out_path.open("r", newline="") as existing:
            reader = csv.reader(existing)
            headers = next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                base = list(map(int, row[:4]))
                if len(row) >= 10:
                    extra = row[4:8]
                    f_ms = float(row[8]); total_ms = float(row[9])
                else:
                    extra = [None, None, None, None]
                    f_ms = float(row[4]); total_ms = float(row[5])
                s2_vals = [int(v) if v not in ("", "None", None) else None for v in extra]
                seen.add(tuple(base + s2_vals))
                results.append(tuple(base + s2_vals + [f_ms, total_ms]))
        writer_mode = "a"
    else:
        writer_mode = "w"
        seen.clear()

    with out_path.open(writer_mode, newline="") as f:
        writer = csv.writer(f)
        if writer_mode == "w":
            writer.writerow([
                "ko_step", "tile_h", "tile_w", "micro_n",
                "ko_step_s2", "tile_h_s2", "tile_w_s2", "micro_n_s2",
                "fwd_ms", "total_ms",
            ])
        for ko, th, tw, micro, ko_s2, th_s2, tw_s2, micro_s2 in itertools.product(
            ko_vals, th_vals, tw_vals, micro_vals, ko_s2_vals, th_s2_vals, tw_s2_vals, micro_s2_vals
        ):
            key = (ko, th, tw, micro, ko_s2, th_s2, tw_s2, micro_s2)
            if args.resume and key in seen:
                print(f"[sweep] skip ko={ko} tile={th}x{tw} micro={micro} s2=({ko_s2},{th_s2},{tw_s2},{micro_s2}) (cached)")
                continue
            env = {
                "XPU_FUSED_KO_STEP": str(ko),
                "XPU_FUSED_TILE_S1": f"{th}x{tw}",
                "XPU_FUSED_MICRO_N": str(micro),
            }
            if ko_s2 is not None:
                env["XPU_FUSED_KO_STEP_S2"] = str(ko_s2)
            if th_s2 is not None and tw_s2 is not None:
                env["XPU_FUSED_TILE_S2"] = f"{th_s2}x{tw_s2}"
            if micro_s2 is not None:
                env["XPU_FUSED_MICRO_N_S2"] = str(micro_s2)
            try:
                f_ms, total_ms = run_one({"config": args.config}, env, args.dbg_max_steps)
            except Exception as exc:
                print(f"[sweep] skip ko={ko} tile={th}x{tw} micro={micro} s2=({ko_s2},{th_s2},{tw_s2},{micro_s2}): {exc}")
                continue
            writer.writerow([ko, th, tw, micro, ko_s2, th_s2, tw_s2, micro_s2, f_ms, total_ms])
            f.flush()
            print(f"[sweep] ko={ko} tile={th}x{tw} micro={micro} "
                  f"s2=({ko_s2},{th_s2},{tw_s2},{micro_s2}) fwd={f_ms:.1f}ms total={total_ms:.1f}ms")
            results.append((ko, th, tw, micro, ko_s2, th_s2, tw_s2, micro_s2, f_ms, total_ms))

    if results:
        top = sorted(results, key=lambda x: x[-2])[:5]
        print("\n=== Top (by fwd ms) ===")
        for ko, th, tw, micro, ko_s2, th_s2, tw_s2, micro_s2, f_ms, total_ms in top:
            print(f"ko={ko} tile={th}x{tw} micro={micro} "
                  f"s2=({ko_s2},{th_s2},{tw_s2},{micro_s2}) -> fwd={f_ms:.1f}ms total={total_ms:.1f}ms")


if __name__ == "__main__":
    main()
