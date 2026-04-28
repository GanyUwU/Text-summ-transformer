import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime

# -----------------------------
# Technical Gantt Chart Dataset
# -----------------------------
data = [
    {"Task": "Requirements & Features", "Start": "2025-08-01", "End": "2025-08-20"},
    {"Task": "Milestone: Scope Finalized", "Start": "2025-08-25", "End": "2025-08-25"},

    {"Task": "Architecture Blueprint", "Start": "2025-09-01", "End": "2025-09-25"},
    {"Task": "Milestone: Design Approved", "Start": "2025-09-28", "End": "2025-09-28"},

    {"Task": "Core Build", "Start": "2025-10-01", "End": "2025-11-15"},
    {"Task": "Backend Integration", "Start": "2025-11-10", "End": "2025-12-20"},
    {"Task": "Milestone: MVP Ready", "Start": "2025-12-22", "End": "2025-12-22"},

    {"Task": "Bug Fixes & Stabilization", "Start": "2026-01-01", "End": "2026-01-25"},
    {"Task": "Training & Optimization", "Start": "2026-02-01", "End": "2026-02-20"},
    {"Task": "Milestone: Model Converged", "Start": "2026-02-25", "End": "2026-02-25"},

    {"Task": "Benchmarking & Documentation", "Start": "2026-03-01", "End": "2026-03-20"},
    {"Task": "Milestone: Final Delivery", "Start": "2026-03-25", "End": "2026-03-25"},
]

df = pd.DataFrame(data)
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = pd.to_datetime(df["End"])

tasks = df["Task"].tolist()
tasks.reverse()

# -----------------------------
# Figure Setup
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)

# -----------------------------
# Axis Formatting
# -----------------------------
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.tick_top()
ax.tick_params(axis='x', labelsize=11)

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(tasks, fontsize=10)
ax.tick_params(axis='y', pad=20, left=False)

# Grid lines
ax.grid(axis='x', linestyle='--', linewidth=0.6)
ax.grid(axis='y', linestyle=':', linewidth=0.3)

# -----------------------------
# Draw Bars + Milestones
# -----------------------------
for _, row in df.iterrows():
    y = tasks.index(row["Task"])
    start = mdates.date2num(row["Start"])
    end = mdates.date2num(row["End"])

    if row["Start"] == row["End"]:
        # Milestone diamond
        ax.scatter(
            start,
            y,
            marker='D',
            s=70
        )
    else:
        width = end - start
        bar = patches.FancyBboxPatch(
            (start, y - 0.25),
            width,
            0.5,
            boxstyle="round,pad=0.02",
            linewidth=1
        )
        ax.add_patch(bar)

# -----------------------------
# Limits
# -----------------------------
ax.set_xlim(
    mdates.date2num(datetime(2025, 7, 20)),
    mdates.date2num(datetime(2026, 3, 31))
)
ax.set_ylim(-1, len(tasks))

# -----------------------------
# Title
# -----------------------------
plt.title(
    "Technical Project Gantt Chart",
    fontsize=16,
    fontweight='bold',
    pad=30
)

plt.tight_layout()
plt.savefig("technical_gantt_chart.png", dpi=300, bbox_inches='tight')
plt.show()