"""Analyze log files to extract per-model BALROG scores."""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Load BALROG achievements data
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ACHIEVEMENTS_PATH = PROJECT_ROOT / "src" / "scoring" / "achievements.json"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

with open(ACHIEVEMENTS_PATH) as f:
    ACHIEVEMENTS = json.load(f)


def balrog_score(depth: int, xp_level: int) -> float:
    """Calculate BALROG progression (0.0-1.0) from depth and XP level."""
    dlvl_prog = ACHIEVEMENTS.get(f"Dlvl:{depth}", 0.0)
    xp_prog = ACHIEVEMENTS.get(f"Xp:{xp_level}", 0.0)
    return max(dlvl_prog, xp_prog)


def analyze_log(filepath: str) -> dict | None:
    """Extract model, max depth, and max XP level from a single log file.

    Returns dict with keys: model, max_depth, max_xp_level, or None if
    the log has no usable data.
    """
    model = None
    max_depth = 0
    max_xp_level = 0

    model_re = re.compile(r"LLM REQUEST: (\S+)")
    status_re = re.compile(r"Dlvl:(\d+)\s.*?Xp:(\d+)/")

    with open(filepath, errors="replace") as f:
        for line in f:
            if model is None:
                m = model_re.search(line)
                if m:
                    model = m.group(1)

            m = status_re.search(line)
            if m:
                depth = int(m.group(1))
                xp_level = int(m.group(2))
                if depth > max_depth:
                    max_depth = depth
                if xp_level > max_xp_level:
                    max_xp_level = xp_level

    if model is None or (max_depth == 0 and max_xp_level == 0):
        return None

    return {
        "model": model,
        "max_depth": max_depth,
        "max_xp_level": max_xp_level,
    }


def main():
    logs_dir = LOGS_DIR
    if len(sys.argv) > 1:
        logs_dir = Path(sys.argv[1])

    if not logs_dir.is_dir():
        print(f"Logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        print(f"No log files found in {logs_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect per-run results grouped by model
    model_runs: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for lf in log_files:
        result = analyze_log(lf)
        if result is None:
            skipped += 1
            continue
        result["balrog"] = balrog_score(result["max_depth"], result["max_xp_level"])
        result["file"] = lf.name
        model_runs[result["model"]].append(result)

    total_runs = sum(len(runs) for runs in model_runs.values())
    print(f"Analyzed {total_runs} runs across {len(model_runs)} model(s) "
          f"({skipped} logs skipped)\n")

    # Build rows sorted by avg BALROG descending
    rows = []
    for model in model_runs:
        runs = model_runs[model]
        n = len(runs)
        depths = [r["max_depth"] for r in runs]
        xp_levels = [r["max_xp_level"] for r in runs]
        balrog_scores = [r["balrog"] for r in runs]
        best = max(runs, key=lambda r: r["balrog"])
        rows.append({
            "model": model,
            "runs": n,
            "avg_depth": sum(depths) / n,
            "max_depth": max(depths),
            "avg_xp": sum(xp_levels) / n,
            "max_xp": max(xp_levels),
            "avg_balrog": sum(balrog_scores) / n,
            "max_balrog": max(balrog_scores),
            "best_file": best["file"],
        })
    rows.sort(key=lambda r: r["avg_balrog"], reverse=True)

    # Print markdown table
    print("| Model | Runs | Avg Depth | Max Depth | Avg XP | Max XP | Avg BALROG | Max BALROG | Best Run |")
    print("|-------|-----:|----------:|----------:|-------:|-------:|-----------:|-----------:|----------|")
    for r in rows:
        print(f"| {r['model']} | {r['runs']} | {r['avg_depth']:.1f} | {r['max_depth']} "
              f"| {r['avg_xp']:.1f} | {r['max_xp']} "
              f"| {r['avg_balrog'] * 100:.2f}% | {r['max_balrog'] * 100:.2f}% "
              f"| {r['best_file']} |")


if __name__ == "__main__":
    main()
