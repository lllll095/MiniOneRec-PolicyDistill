#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

required = [
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "requirements.txt",
    "src/__init__.py",
    "src/policy_distill.py",
    "src/data.py",
    "src/sft_train.py",
    "src/grpo_train.py",
    "src/LogitProcessor.py",
    "src/sasrec.py",
    "src/evaluate.py",
    "src/calc.py",
    "scripts/train_sft.sh",
    "scripts/train_grpo.sh",
    "scripts/train_kd.sh",
    "scripts/train_opd.sh",
    "scripts/evaluate.sh",
    "results/main_results.csv",
    "results/teacher_student_diagnostics.csv",
    "assets/main_results.png",
    "assets/hr_curve.png",
    "assets/policy_entropy.png",
]

missing = []
print("Repository:", ROOT)
print()

for rel in required:
    ok = (ROOT / rel).exists()
    print(f"[{'OK' if ok else 'MISSING':7s}] {rel}")
    if not ok:
        missing.append(rel)

print("\nLarge-file scan (>20 MB)")
large = []
for p in ROOT.rglob("*"):
    if p.is_file() and ".git" not in p.parts and p.stat().st_size > 20 * 1024 * 1024:
        large.append((p.relative_to(ROOT), p.stat().st_size))

if large:
    for rel, size in sorted(large):
        print(f"{size / 1024**2:8.1f} MB  {rel}")
else:
    print("none")

if missing:
    print("\nFAIL: required files are missing.")
    sys.exit(1)

print("\nPASS: required project files are present.")
