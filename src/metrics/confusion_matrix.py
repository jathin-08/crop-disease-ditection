"""
Tomato Disease AI - Confusion Matrix Visualizer
Generates and exports normalized and raw confusion matrix heatmaps.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = True,
    title: str = "Tomato Leaf Disease - Confusion Matrix"
) -> Path:
    """Renders and saves a stylish confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(11, 9), dpi=200)
    sns.set_theme(style="white")

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        matrix_to_plot = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"
    else:
        matrix_to_plot = cm.astype(int)
        fmt = "d"

    clean_names = [n.replace("tomato_", "").replace("_", " ").title() for n in class_names]

    ax = sns.heatmap(
        matrix_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Greens" if normalize else "Blues",
        xticklabels=clean_names,
        yticklabels=clean_names,
        cbar=True,
        linewidths=0.6,
        linecolor="#e0e0e0"
    )

    plt.title(title, fontsize=14, weight="bold", pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=11, weight="bold")
    plt.ylabel("True Disease Class", fontsize=11, weight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path
