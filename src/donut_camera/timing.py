import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    synchronize()
    start = time.perf_counter()
    result = call()
    synchronize()
    return result, (time.perf_counter() - start) * 1_000


def summarize_ms(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999) - 1))
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def benchmark(
    call: Callable[[], Any], *, warmups: int = 10, repetitions: int = 30
) -> dict:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    for _ in range(warmups):
        call()
    synchronize()
    values = [timed(call)[1] for _ in range(repetitions)]
    return {"raw_ms": values, **summarize_ms(values)}


def cuda_memory(call: Callable[[], Any]) -> dict | None:
    if not torch.cuda.is_available():
        call()
        return None
    synchronize()
    start_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    call()
    synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    return {
        "start_allocated_mb": start_allocated / 1024**2,
        "peak_allocated_mb": peak_allocated / 1024**2,
        "incremental_peak_mb": (peak_allocated - start_allocated) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }
