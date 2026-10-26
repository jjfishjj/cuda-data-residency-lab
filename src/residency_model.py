"""Estimate host/device transfer cost and recommend a CUDA transfer strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TransferPlan:
    payload_mb: float
    iterations: int
    transfers_per_iteration: int
    bandwidth_gbps: float
    transfer_ms: float
    recommendation: str


def build_plan(
    payload_mb: float,
    iterations: int,
    transfers_per_iteration: int = 2,
    bandwidth_gbps: float = 24.0,
) -> TransferPlan:
    if payload_mb <= 0 or iterations <= 0 or transfers_per_iteration < 0 or bandwidth_gbps <= 0:
        raise ValueError("payload, iterations and bandwidth must be positive; transfers cannot be negative")
    total_gb = payload_mb / 1024 * iterations * transfers_per_iteration
    transfer_ms = total_gb / bandwidth_gbps * 1000
    if transfers_per_iteration == 0:
        recommendation = "資料已常駐 GPU；維持 device-resident pipeline"
    elif transfer_ms >= 100:
        recommendation = "優先使用 pinned memory、批次傳輸與 CUDA streams"
    else:
        recommendation = "傳輸成本可控；仍應避免細碎同步與逐筆拷貝"
    return TransferPlan(payload_mb, iterations, transfers_per_iteration, bandwidth_gbps, transfer_ms, recommendation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-mb", type=float, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--transfers", type=int, default=2)
    parser.add_argument("--bandwidth-gbps", type=float, default=24)
    args = parser.parse_args()
    print(json.dumps(asdict(build_plan(args.payload_mb, args.iterations, args.transfers, args.bandwidth_gbps)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
