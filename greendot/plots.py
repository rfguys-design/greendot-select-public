"""Draw a chart from the table the model extracted. OWNER: James

The model only READS the table (into numbers); drawing is deterministic code,
so the chart always matches the extracted values exactly.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")  # draw to a file; no window, safe from any thread
import matplotlib.pyplot as plt

from greendot.config import CONFIG


def render_chart(chart):
    """chart: dict from RESPONSE_SCHEMA. Returns the PNG path, or None if there is no data."""
    x = chart["x_values"]
    series = [s for s in chart["series"] if s["values"]]
    if not x or not series:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    kind = chart["chart_type"]

    if kind == "pie":
        s = series[0]
        ax.pie(s["values"][:len(x)], labels=x[:len(s["values"])], autopct="%1.0f%%", startangle=90)
        ax.axis("equal")
    elif kind == "bar":
        width = 0.8 / len(series)
        for i, s in enumerate(series):
            n = min(len(x), len(s["values"]))
            positions = [j + (i - (len(series) - 1) / 2) * width for j in range(n)]
            ax.bar(positions, s["values"][:n], width, label=s["name"])
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x)
    else:
        for s in series:
            n = min(len(x), len(s["values"]))
            if kind == "scatter":
                ax.scatter(x[:n], s["values"][:n], label=s["name"], s=40)
            else:
                ax.plot(x[:n], s["values"][:n], marker="o", label=s["name"])

    if kind != "pie":
        ax.set_xlabel(chart["x_label"])
        ax.set_ylabel(chart["y_label"])
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if len(series) > 1:
            ax.legend(frameon=False)
        if max(len(str(v)) for v in x) > 6:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(chart["title"])
    fig.tight_layout()

    path = os.path.join(CONFIG["plot_dir"], time.strftime("plot_%Y%m%d_%H%M%S.png"))
    fig.savefig(path)
    plt.close(fig)
    return path
