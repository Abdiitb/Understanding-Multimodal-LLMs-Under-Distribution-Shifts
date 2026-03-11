import json
import matplotlib.pyplot as plt
import numpy as np

# Load the EMID data
with open("emid_by_shift_severity_natural.json", "r") as f:
    data = json.load(f)

# Compute average EMID for each shift scenario
avg_emid = {}
for scenario, values in data.items():
    avg_emid[scenario] = {
        "average": np.mean(list(values.values())),
        "std": np.std(list(values.values())),
        "num_datasets": len(values),
        "individual_values": values,
    }

# Save averages to JSON
avg_emid_save = {k: v["average"] for k, v in avg_emid.items()}
with open("avg_emid_by_shift_severity.json", "w") as f:
    json.dump(avg_emid_save, f, indent=4)
print("Saved average EMID values to avg_emid_by_shift_severity.json")
print(json.dumps(avg_emid_save, indent=4))

# --- Plot ---
scenarios = list(avg_emid.keys())
means = [avg_emid[s]["average"] for s in scenarios]
stds = [avg_emid[s]["std"] for s in scenarios]

fig, ax = plt.subplots(figsize=(8, 5))

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.bar(scenarios, means, yerr=stds, capsize=6, color=colors[:len(scenarios)],
              edgecolor="black", linewidth=0.8, alpha=0.85)

# Add value labels on bars
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xlabel("Shift Severity", fontsize=13)
ax.set_ylabel("Average EMID", fontsize=13)
ax.set_title("Average EMID by Shift Severity (Natural Shift)", fontsize=14, fontweight="bold")
ax.tick_params(axis="both", labelsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("emid_by_shift_severity_plot.png", dpi=200)
plt.show()
print("Plot saved to emid_by_shift_severity_plot.png")
