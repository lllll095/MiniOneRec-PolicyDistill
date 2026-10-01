from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS / "main_results.csv")

order = ["SFT", "SFT-GRPO", "Gold-KD", "OPD", "OPD-GRPO", "Teacher-1.5B"]
pretty = {
    "SFT": "0.5B SFT",
    "SFT-GRPO": "0.5B GRPO",
    "Gold-KD": "0.5B Gold-KD",
    "OPD": "0.5B OPD",
    "OPD-GRPO": "0.5B OPD->GRPO",
    "Teacher-1.5B": "1.5B Teacher",
}

df = df.set_index("method").loc[order].reset_index()
df["display"] = df["method"].map(pretty)

# Main metric bars
fig, ax = plt.subplots(figsize=(10, 5.5))
x = list(range(len(df)))
width = 0.24
for j, metric in enumerate(["HR@10", "NDCG@10", "HR@50"]):
    offset = (j - 1) * width
    ax.bar([i + offset for i in x], df[metric], width=width, label=metric)
ax.set_xticks(x)
ax.set_xticklabels(df["display"], rotation=20, ha="right")
ax.set_ylabel("Metric")
ax.set_title("MiniOneRec Policy Distillation: Main Results")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(ASSETS / "main_results.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# HR@K curves
ks = [1, 3, 5, 10, 20, 50]
fig, ax = plt.subplots(figsize=(8, 5.5))
for _, row in df.iterrows():
    ax.plot(ks, [row[f"HR@{k}"] for k in ks], marker="o", label=row["display"])
ax.set_xlabel("K")
ax.set_ylabel("Hit Rate")
ax.set_title("HR@K across Training Strategies")
ax.set_xticks(ks)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ASSETS / "hr_curve.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Entropy diagnostic
diag = pd.read_csv(RESULTS / "teacher_student_diagnostics.csv")
fig, ax = plt.subplots(figsize=(7.5, 5))
x = list(range(len(diag)))
width = 0.36
ax.bar([i - width/2 for i in x], diag["student_entropy"], width=width, label="0.5B Student")
ax.bar([i + width/2 for i in x], diag["teacher_entropy"], width=width, label="1.5B Teacher")
ax.set_xticks(x)
ax.set_xticklabels(diag["level"])
ax.set_ylabel("Legal-action entropy")
ax.set_title("Teacher Policy Is Sharper than Student Policy")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(ASSETS / "policy_entropy.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Figures written to:", ASSETS)
