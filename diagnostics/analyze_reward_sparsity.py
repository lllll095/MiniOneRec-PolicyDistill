#!/usr/bin/env python3
import argparse
import ast
from pathlib import Path
import numpy as np

KEYS = [
    "rewards/rule_reward",
    "rewards/ndcg_rule_reward",
    "reward",
    "reward_std",
    "grad_norm",
    "kl",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_file")
    args = p.parse_args()

    rows = []
    for raw in Path(args.log_file).read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)

    print("parsed rows:", len(rows))

    for key in KEYS:
        x = np.asarray([r[key] for r in rows if key in r], dtype=float)
        if len(x) == 0:
            continue
        print(
            f"{key:28s} "
            f"n={len(x):4d} "
            f"mean={x.mean():.6f} "
            f"median={np.median(x):.6f} "
            f"min={x.min():.6f} "
            f"max={x.max():.6f}"
        )

if __name__ == "__main__":
    main()
