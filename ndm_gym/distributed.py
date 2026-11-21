"""Synchronous data parallel helpers for NDM-Gym training.

The training algorithms call modules through custom compiled functions, so a
``DistributedDataParallel`` wrapper is not a reliable synchronization boundary.
Instead, every rank owns a simulator and replay buffer and explicitly averages
gradients after backward and before clipping/optimizer steps.

The module is inert unless launched by ``torchrun`` with ``WORLD_SIZE > 1``.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


_rank = 0
_world_size = 1
_local_rank = 0
_initialized_here = False


def initialize(timeout_minutes: int = 30) -> str:
    """Initialize the process group from ``torchrun``'s environment.

    Returns the CUDA device for this process. A normal Python launch keeps the
    historical ``cuda:0`` single-GPU path.
    """

    global _rank, _world_size, _local_rank, _initialized_here
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _rank = int(os.environ.get("RANK", "0"))
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if _world_size <= 1:
        _rank, _world_size, _local_rank = 0, 1, 0
        return "cuda:0"

    backend = os.environ.get("NDM_GYM_DIST_BACKEND", "nccl")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU training needs CUDA")
        torch.cuda.set_device(_local_rank)
        device = f"cuda:{_local_rank}"
    elif backend == "gloo":  # CPU process-group tests only; the simulator still requires CUDA.
        device = "cpu"
    else:
        raise ValueError(f"unsupported NDM_GYM_DIST_BACKEND={backend!r}")
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
        _initialized_here = True
    return device


def shutdown() -> None:
    """Destroy a process group created by :func:`initialize`."""

    global _initialized_here
    if _initialized_here and dist.is_initialized():
        dist.destroy_process_group()
    _initialized_here = False


def enabled() -> bool:
    return _world_size > 1


def rank() -> int:
    return _rank


def local_rank() -> int:
    return _local_rank


def world_size() -> int:
    return _world_size


def is_main() -> bool:
    return _rank == 0


def barrier() -> None:
    if enabled():
        dist.barrier()


@torch.no_grad()
def broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Broadcast parameters and persistent buffers from ``src``."""

    if not enabled():
        return
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


@torch.no_grad()
def average_gradients(parameters: Iterable[nn.Parameter]) -> None:
    """Average existing dense gradients over ranks.

    All ranks execute the same WM/policy/critic graph, so the set, order and
    shapes of gradients are identical. One coalesced collective avoids a
    separate NCCL launch for every parameter tensor.
    """

    if not enabled():
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    for grad in grads:
        if grad.is_sparse:
            raise RuntimeError("NDM-Gym distributed training does not support sparse gradients")
    dist.all_reduce_coalesced(grads, op=dist.ReduceOp.SUM)
    scale = 1.0 / _world_size
    for grad in grads:
        grad.mul_(scale)


@torch.no_grad()
def global_batch_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global ``(count, mean, variance)`` for rows of ``x``.

    The reduction uses sums and squared sums, so every rank applies exactly the
    same RunningNorm update while retaining only a local replay buffer.
    """

    x = x.reshape(-1, x.shape[-1]).float()
    count = x.new_tensor(float(x.shape[0]))
    total = x.sum(0)
    square_total = (x * x).sum(0)
    if enabled():
        packed = torch.cat((count.view(1), total, square_total))
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        count = packed[0]
        dim = x.shape[-1]
        total = packed[1 : 1 + dim]
        square_total = packed[1 + dim :]
    mean = total / count.clamp(min=1)
    var = (square_total / count.clamp(min=1) - mean.square()).clamp(min=0)
    return count, mean, var


@torch.no_grad()
def mean_scalar(value: float | torch.Tensor, device: torch.device | str, *, ignore_nan: bool = True) -> float:
    """Mean one scalar metric across ranks, optionally ignoring NaNs."""

    value_t = torch.as_tensor(value, dtype=torch.float64, device=device)
    if not enabled():
        return float(value_t)
    valid = torch.isfinite(value_t) if ignore_nan else torch.ones_like(value_t, dtype=torch.bool)
    pair = torch.stack((torch.where(valid, value_t, 0.0), valid.to(torch.float64)))
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    if pair[1] == 0:
        return float("nan")
    return float(pair[0] / pair[1])


@torch.no_grad()
def sum_pair(first: float | torch.Tensor, second: float | torch.Tensor, device: torch.device | str) -> tuple[float, float]:
    """Sum two scalar counters over ranks with one collective."""

    pair = torch.as_tensor((first, second), dtype=torch.float64, device=device)
    if enabled():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float(pair[0]), float(pair[1])


@torch.no_grad()
def rank0_scalar(value: float, device: torch.device | str) -> float:
    """Broadcast a scalar control value chosen by rank 0."""

    value_t = torch.tensor(float(value) if is_main() else 0.0, dtype=torch.float64, device=device)
    if enabled():
        dist.broadcast(value_t, src=0)
    return float(value_t)
