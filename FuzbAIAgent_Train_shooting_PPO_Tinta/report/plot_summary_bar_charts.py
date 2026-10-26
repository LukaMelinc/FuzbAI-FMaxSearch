#!/usr/bin/env python3
"""Generate bar-chart comparisons from evaluation *.summary.json files.

Examples:
  python3 report/plot_summary_bar_charts.py \
    --summary-glob 'report/*.summary.json' \
    --output-dir report/figures

  python3 report/plot_summary_bar_charts.py \
    --summary-file report/eval_strel-sirina3.summary.json \
    --summary-file report/eval_strel-sirina2.summary.json \
    --title 'Primerjava strelov'
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create bar charts from evaluation summary JSON files."
    )
    parser.add_argument(
        "--summary-file",
        action="append",
        default=[],
        help="Explicit path to a .summary.json file (repeatable).",
    )
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=["report/*.summary.json", "evaluacije/*.summary*.json"],
        help="Glob pattern for summary files (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        default="report/figures",
        help="Directory for generated images.",
    )
    parser.add_argument(
        "--prefix",
        default="summary_bar",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--title",
        default="Primerjava evalvacij",
        help="Main title used in charts.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("name", "success_rate"),
        default="name",
        help="Sort compared runs by name or success rate.",
    )
    parser.add_argument(
        "--bar-mode",
        choices=("auto", "outcome", "kick_stats"),
        default="auto",
        help=(
            "Which 3 bars to draw: outcome -> success/partial/failure, "
            "kick_stats -> mean/min/max, auto -> kick_stats if available in all files."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show charts interactively in addition to saving.",
    )
    parser.add_argument(
        "--name-override",
        action="append",
        default=[],
        help=(
            "Override run name for a specific file. Format: 'file=Display name'. "
            "The file part can be either full path or basename. Repeatable."
        ),
    )
    return parser.parse_args()


def collect_files(globs: List[str], explicit: List[str]) -> List[Path]:
    files = {Path(p).resolve() for p in explicit}
    for pattern in globs:
        for path in Path(".").glob(pattern):
            if path.is_file():
                files.add(path.resolve())
    return sorted(files)


def build_label(path: Path) -> str:
    name = path.name
    if name.endswith(".summary.json"):
        return name[: -len(".summary.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def parse_name_overrides(items: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --name-override '{raw}'. Use format: file=Display name"
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid --name-override '{raw}'. Use format: file=Display name"
            )
        mapping[key] = value
    return mapping


def load_summary(path: Path, name_overrides: Dict[str, str]) -> Dict[str, float]:
    data = json.loads(path.read_text())
    episodes = int(data.get("episodes", 0))
    if episodes <= 0:
        raise ValueError(f"Invalid episodes in {path}")

    success = int(data.get("success", 0))
    partial = int(data.get("partial_success", 0))
    failure = int(data.get("failure", 0))

    if success + partial + failure != episodes:
        # Keep script tolerant to older logs with slight mismatches.
        episodes = max(episodes, success + partial + failure)

    default_name = str(data.get("name") or data.get("label") or build_label(path))
    resolved_name = name_overrides.get(str(path), name_overrides.get(path.name, default_name))

    return {
        "label": resolved_name,
        "name": resolved_name,
        "file": str(path),
        "episodes": episodes,
        "success": success,
        "partial_success": partial,
        "failure": failure,
        "mean": float(data.get("mean", float("nan"))),
        "min": float(data.get("min", float("nan"))),
        "max": float(data.get("max", float("nan"))),
        "success_rate": float(data.get("success_rate", success / episodes if episodes else 0.0)),
    }


def sort_rows(rows: List[Dict[str, float]], mode: str) -> List[Dict[str, float]]:
    if mode == "success_rate":
        return sorted(rows, key=lambda r: r["success_rate"], reverse=True)
    return sorted(rows, key=lambda r: r["name"])


def resolve_bar_mode(rows: List[Dict[str, float]], requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    has_kick_stats = all(
        math.isfinite(float(r.get("mean", float("nan"))))
        and math.isfinite(float(r.get("min", float("nan"))))
        and math.isfinite(float(r.get("max", float("nan"))))
        for r in rows
    )
    return "kick_stats" if has_kick_stats else "outcome"


def plot_grouped_counts(
    rows: List[Dict[str, float]],
    out_path: Path,
    title: str,
    show: bool,
    bar_mode: str,
) -> None:
    labels = [r["name"] for r in rows]

    if bar_mode == "kick_stats":
        success = np.array([r["mean"] for r in rows], dtype=float)
        partial = np.array([r["min"] for r in rows], dtype=float)
        failure = np.array([r["max"] for r in rows], dtype=float)
        y_label = "Kontaktna sila [N]"
        l1, l2, l3 = "mean", "min", "max"
        show_percent = False
    else:
        success = np.array([r["success"] for r in rows], dtype=float)
        partial = np.array([r["partial_success"] for r in rows], dtype=float)
        failure = np.array([r["failure"] for r in rows], dtype=float)
        y_label = "Število epizod"
        l1, l2, l3 = "Zadetek", "Delni uspeh", "Zgrešitev"
        show_percent = True

    episodes = np.array([r["episodes"] for r in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(rows)), 5.6))
    b1 = ax.bar(x - width, success, width=width, label=l1, color="#2E8B57", edgecolor="black", linewidth=0.35)
    b2 = ax.bar(x, partial, width=width, label=l2, color="#E6A700", edgecolor="black", linewidth=0.35)
    b3 = ax.bar(x + width, failure, width=width, label=l3, color="#C0392B", edgecolor="black", linewidth=0.35)

    # Add both absolute count and percentage of total episodes for each run.
    #for bars, counts in ((b1, success), (b3, failure)):
    for bars, counts in ((b1, success), (b2, partial), (b3, failure)):
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            denom = episodes[idx] if episodes[idx] > 0 else 1.0
            pct = counts[idx] / denom
            if show_percent:
                text = f"{int(h)}\n({pct:.1%})"
            else:
                text = f"{h:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.6,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x, labels, rotation=25)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    #ymax = max(np.max(success), np.max(partial), np.max(failure)) if len(rows) else 0
    ymax = max(np.max(success), np.max(partial), np.max(failure)) if len(rows) else 0
    ax.set_ylim(0, ymax * 1.22 + 2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show(block=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    files = collect_files(args.summary_glob, args.summary_file)
    if not files:
        raise SystemExit("No summary files found. Use --summary-file or adjust --summary-glob.")

    try:
        name_overrides = parse_name_overrides(args.name_override)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    normalized_overrides: Dict[str, str] = {}
    for key, value in name_overrides.items():
        key_path = Path(key)
        if key_path.is_absolute() or "/" in key or key.startswith("."):
            normalized_overrides[str(key_path.resolve())] = value
        normalized_overrides[key_path.name] = value

    rows = [load_summary(path, normalized_overrides) for path in files]
    rows = sort_rows(rows, args.sort_by)
    resolved_mode = resolve_bar_mode(rows, args.bar_mode)

    if resolved_mode == "kick_stats":
        missing = [r["file"] for r in rows if not all(math.isfinite(float(r[k])) for k in ("mean", "min", "max"))]
        if missing:
            raise SystemExit(
                "Missing mean/min/max in files:\n- " + "\n- ".join(missing)
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_counts = output_dir / f"{args.prefix}_grouped_counts.png"

    plot_grouped_counts(rows, out_counts, args.title, args.show, resolved_mode)

    print("Generated bar chart:")
    print(f"- {out_counts}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
