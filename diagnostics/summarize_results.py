#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "main_results.csv").set_index("method")

for method in [
    "SFT",
    "SFT-GRPO",
    "Gold-KD",
    "OPD",
    "OPD-GRPO",
    "Teacher-1.5B",
]:
    r = df.loc[method]
    print(
        f"{method:14s} "
        f"HR@10={r['HR@10']:.5f} "
        f"NDCG@10={r['NDCG@10']:.5f} "
        f"HR@50={r['HR@50']:.5f}"
    )

sft = df.loc["SFT"]
opd = df.loc["OPD"]
teacher = df.loc["Teacher-1.5B"]

gap = (
    (opd["HR@50"] - sft["HR@50"])
    / (teacher["HR@50"] - sft["HR@50"])
)

print()
print(f"OPD recovers {100 * gap:.1f}% of the SFT->teacher HR@50 gap.")
