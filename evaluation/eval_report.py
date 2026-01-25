"""
Generate combined evaluation report (open-loop + closed-loop).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def load_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        print(f"⚠️ 未找到文件: {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_metrics_table(open_loop: Dict, closed_loop: Dict) -> str:
    open_metrics = open_loop.get("metrics", {})
    closed_metrics = closed_loop.get("metrics_mean", closed_loop.get("metrics", {}))

    rows = [
        ("open_loop_ade", open_metrics.get("ade")),
        ("open_loop_fde", open_metrics.get("fde")),
        ("route_completion", closed_metrics.get("route_completion")),
        ("collision_count", closed_metrics.get("collision_count")),
        ("red_light_violations", closed_metrics.get("red_light_violations")),
        ("off_road_percentage", closed_metrics.get("off_road_percentage")),
        ("infraction_score", closed_metrics.get("infraction_score")),
    ]

    lines = [
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    for metric, value in rows:
        lines.append(f"| {metric} | {format_float(value)} |")
    return "\n".join(lines)


def build_csv(open_loop: Dict, closed_loop: Dict) -> str:
    open_metrics = open_loop.get("metrics", {})
    closed_metrics = closed_loop.get("metrics_mean", closed_loop.get("metrics", {}))

    rows = {
        "open_loop_ade": open_metrics.get("ade"),
        "open_loop_fde": open_metrics.get("fde"),
        "route_completion": closed_metrics.get("route_completion"),
        "collision_count": closed_metrics.get("collision_count"),
        "red_light_violations": closed_metrics.get("red_light_violations"),
        "off_road_percentage": closed_metrics.get("off_road_percentage"),
        "infraction_score": closed_metrics.get("infraction_score"),
    }
    lines = ["metric,value"]
    for key, value in rows.items():
        lines.append(f"{key},{value if value is not None else ''}")
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成迭代评估报表")
    parser.add_argument("--open-loop", type=str, default=None, help="open_loop_eval 输出 JSON")
    parser.add_argument("--closed-loop", type=str, default=None, help="closed_loop_sim summary.json")
    parser.add_argument("--output", type=str, default="reports/eval_report.md")
    parser.add_argument("--format", type=str, choices=["md", "csv"], default="md")
    return parser


def main():
    args = build_argparser().parse_args()
    open_loop = load_json(args.open_loop)
    closed_loop = load_json(args.closed_loop)

    if args.format == "csv":
        content = build_csv(open_loop, closed_loop)
    else:
        content = build_metrics_table(open_loop, closed_loop)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ 报表已生成: {output_path}")


if __name__ == "__main__":
    main()
