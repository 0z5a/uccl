"""Shared, deterministic style for UCCL Lite's data figures."""

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#205E9E"
TEAL = "#148D91"
AMBER = "#BF6C36"
PURPLE = "#8064A6"
BASELINE = "#8999A8"
SECOND_BASELINE = "#B3BEC8"
INK = "#172A3A"
MUTED = "#607486"
GRID = "#E6EDF2"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "uccl-lite-figures-v2",
        }
    )


def heading(fig, title: str, subtitle: str, left: float = 0.095) -> None:
    fig.text(left, 0.965, title, va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(left, 0.89, subtitle, va="top", fontsize=10.5, color=MUTED)


def clean_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(axis="both", which="both", length=0, pad=7)


def save(fig, name: str, preview: bool) -> None:
    output = ROOT / "docs/assets" / f"{name}.svg"
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    output.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    if preview:
        folder = ROOT / "build/figures"
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=160, facecolor="white")
        fig.savefig(
            folder / f"{name}.pdf",
            metadata={"CreationDate": None, "ModDate": None},
            facecolor="white",
        )
    plt.close(fig)
    print(output.relative_to(ROOT))
