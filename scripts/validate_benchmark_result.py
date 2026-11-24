#!/usr/bin/env python3
"""Small dependency-free validator for the shared benchmark result contract."""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path


def validate(result: dict) -> list[str]:
    errors: list[str] = []
    required = ("schema_version", "benchmark_id", "run_id", "timestamp_utc", "git", "environment", "parameters", "measurements")
    for key in required:
        if key not in result:
            errors.append(f"missing top-level field: {key}")
    if result.get("schema_version") != "1.0.0": errors.append("schema_version must be 1.0.0")
    try: datetime.fromisoformat(result.get("timestamp_utc", "").replace("Z", "+00:00"))
    except ValueError: errors.append("timestamp_utc is not RFC3339 date-time")
    if not isinstance(result.get("git"), dict) or not {"commit", "dirty"} <= result["git"].keys(): errors.append("git must contain commit and dirty")
    env = result.get("environment", {})
    if not isinstance(env, dict) or not {"os", "architecture", "python", "software", "gpus"} <= env.keys(): errors.append("environment is incomplete")
    for index, measurement in enumerate(result.get("measurements", [])):
        prefix = f"measurements[{index}]"
        for key in ("case_id", "backend", "status", "unit", "input_rows", "warmup_iterations", "iterations_per_sample", "samples", "statistics"):
            if key not in measurement: errors.append(f"{prefix} missing {key}")
        if measurement.get("unit") != "seconds": errors.append(f"{prefix}.unit must be seconds")
        if measurement.get("status") not in ("completed", "skipped", "failed"): errors.append(f"{prefix}.status invalid")
        if not isinstance(measurement.get("samples"), list) or any(not isinstance(x, (int, float)) or x < 0 for x in measurement.get("samples", [])): errors.append(f"{prefix}.samples invalid")
        stats = measurement.get("statistics", {})
        if not isinstance(stats, dict) or not {"median", "mean", "stdev", "min", "max"} <= stats.keys(): errors.append(f"{prefix}.statistics incomplete")
    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("-")
    data = json.load(sys.stdin if str(path) == "-" else path.open())
    errors = validate(data)
    if errors:
        for error in errors: print(error, file=sys.stderr)
        return 1
    print("valid benchmark result: schema_version=1.0.0")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
