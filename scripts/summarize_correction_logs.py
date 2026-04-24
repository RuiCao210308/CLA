"""Summarize OpenVLA LIBERO correction ablation logs.

Usage:
    python scripts/summarize_correction_logs.py experiments/logs/*.txt
    python scripts/summarize_correction_logs.py --per_episode experiments/logs/*.txt
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
        "episode_rows": _episode_rows(path, successes, metrics),
        "episodes": len(successes),
        "success_rate": mean(successes) if successes else None,
        "total_success_rate": total_success_rate,
        "avg_action_delta_norm": _mean_metric(metrics, "avg_action_delta_norm"),
        "gripper_flip_count": _mean_metric(metrics, "gripper_flip_count"),
        "held_gripper_flip_count": _mean_metric(metrics, "held_gripper_flip_count"),
        "stagnation_triggers": _mean_metric(metrics, "stagnation_triggers"),
        "conditional_correction_steps": _mean_metric(metrics, "conditional_correction_steps"),
        "conditional_correction_ratio": _mean_metric(metrics, "conditional_correction_ratio"),
        "max_consecutive_corrections_used": _mean_metric(metrics, "max_consecutive_corrections_used"),
    }


def _episode_rows(path, successes, metrics):
    rows = []
    for episode_idx, success in enumerate(successes, start=1):
        metric = metrics[episode_idx - 1] if episode_idx - 1 < len(metrics) else {}
        rows.append(
            {
                "run": path.stem,
                "episode": episode_idx,
                "success": success,
                "avg_action_delta_norm": _metric_value(metric, "avg_action_delta_norm"),
                "gripper_flip_count": _metric_value(metric, "gripper_flip_count"),
                "held_gripper_flip_count": _metric_value(metric, "held_gripper_flip_count"),
                "stagnation_triggers": _metric_value(metric, "stagnation_triggers"),
                "conditional_correction_steps": _metric_value(metric, "conditional_correction_steps"),
                "conditional_correction_ratio": _metric_value(metric, "conditional_correction_ratio"),
                "max_consecutive_corrections_used": _metric_value(metric, "max_consecutive_corrections_used"),
                "trigger_reason_counts": metric.get("trigger_reason_counts", "-"),
            }
        )
    return rows


def _metric_value(metric, key):
    if key not in metric:
        return None
    try:
        return float(metric[key])
    except ValueError:
        return None


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
    parser.add_argument("--per_episode", action="store_true", help="Print one row per episode instead of run averages.")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [parse_log(path) for path in args.logs]
    rows.sort(key=lambda row: row["path"].name)

    if args.per_episode:
        print_per_episode(rows)
        return

    header = [
        "run",
        "episodes",
        "success_rate",
        "avg_action_delta_norm",
        "gripper_flip_count",
        "held_gripper_flip_count",
        "stagnation_triggers",
        "conditional_correction_steps",
        "conditional_correction_ratio",
        "max_consecutive_corrections_used",
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
                    fmt(row["conditional_correction_steps"]),
                    fmt(row["conditional_correction_ratio"]),
                    fmt(row["max_consecutive_corrections_used"]),
                ]
            )
        )


def print_per_episode(rows) -> None:
    header = [
        "run",
        "episode",
        "success",
        "avg_action_delta_norm",
        "gripper_flip_count",
        "held_gripper_flip_count",
        "stagnation_triggers",
        "conditional_correction_steps",
        "conditional_correction_ratio",
        "max_consecutive_corrections_used",
        "trigger_reason_counts",
    ]
    print("\t".join(header))
    for row in rows:
        for episode_row in row["episode_rows"]:
            print(
                "\t".join(
                    [
                        episode_row["run"],
                        str(episode_row["episode"]),
                        str(episode_row["success"]),
                        fmt(episode_row["avg_action_delta_norm"]),
                        fmt(episode_row["gripper_flip_count"]),
                        fmt(episode_row["held_gripper_flip_count"]),
                        fmt(episode_row["stagnation_triggers"]),
                        fmt(episode_row["conditional_correction_steps"]),
                        fmt(episode_row["conditional_correction_ratio"]),
                        fmt(episode_row["max_consecutive_corrections_used"]),
                        episode_row["trigger_reason_counts"],
                    ]
                )
            )


if __name__ == "__main__":
    main()
