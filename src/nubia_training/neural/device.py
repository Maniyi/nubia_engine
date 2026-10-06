"""Explicit, testable device selection without import-time accelerator work."""

from __future__ import annotations

from collections.abc import Callable

import torch

from nubia_training.neural.errors import DeviceUnavailableError

DeviceChoice = str


def select_device(
    choice: DeviceChoice,
    *,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
) -> torch.device:
    """Resolve cpu/mps/cuda/auto, preferring CUDA then MPS for auto."""

    if choice not in {"cpu", "mps", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, mps, cuda, auto")
    cuda_check = torch.cuda.is_available if cuda_available is None else cuda_available
    mps_check = (
        torch.backends.mps.is_available if mps_available is None else mps_available
    )
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not cuda_check():
            raise DeviceUnavailableError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if choice == "mps":
        if not mps_check():
            raise DeviceUnavailableError("MPS was requested but is unavailable")
        return torch.device("mps")
    if cuda_check():
        return torch.device("cuda")
    if mps_check():
        return torch.device("mps")
    return torch.device("cpu")
