"""Summarize OpenVLA LIBERO correction ablation logs.

Usage:
    python scripts/summarize_correction_logs.py experiments/logs/*.txt
"""

import argparse
import re
from pathlib import Path
from statistics import mean


METRIC_RE = re.compile(r"([A-Za-z_]+)=([^,\n]+)")


def parse_bool(value: str) -> bool:
    return value.strip() == "True"


def parse_float(value: str) -> float:
    return float(value.strip().strip("[]"))


def parse_log(path: Path) -> dict:
    successes = []
    metrics = []
    total_success_rate = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Success:"):
            successes.append(parse_bool(line.split(":", 1)[1]))
        elif line.startswith("Correction debug:"):
            values = {key: value.strip() for key, value in METRIC_RE.findall(line)}
            metrics.append(values)
        elif line.startswith("Current total success rate:"):
            total_success_rate = parse_float(line.split(":", 1)[1])

    return {
        "path": path,
        "episodes": len(successes),
        "success_rate": mean(successes) if successes else None,
        "total_success_rate": total_success_rate,
        "avg_action_delta_norm": _mean_metric(metrics, "avg_action_delta_norm"),
        "gripper_flip_count": _mean_metric(metrics, "gripper_flip_count"),
        "held_gripper_flip_count": _mean_metric(metrics, "held_gripper_flip_count"),
        "stagnation_triggers": _mean_metric(metrics, "stagnation_triggers"),
    }


def _mean_metric(metrics, key):
    values = []
    for metric in metrics:
        if key in metric:
            try:
                values.append(float(metric[key]))
            except ValueError:
                pass
    return mean(values) if values else None


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [parse_log(path) for path in args.logs]
    rows.sort(key=lambda row: row["path"].name)

    header = [
        "run",
        "episodes",
        "success_rate",
        "avg_action_delta_norm",
        "gripper_flip_count",
        "held_gripper_flip_count",
        "stagnation_triggers",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    row["path"].stem,
                    str(row["episodes"]),
                    fmt(row["success_rate"]),
                    fmt(row["avg_action_delta_norm"]),
                    fmt(row["gripper_flip_count"]),
                    fmt(row["held_gripper_flip_count"]),
                    fmt(row["stagnation_triggers"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
