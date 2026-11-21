from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, output: str) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        NDM_GYM_DIST_BACKEND="gloo",
    )
    from ndm_gym import distributed
    from ndm_gym.core.normalizer import RunningNorm

    assert distributed.initialize(timeout_minutes=1) == "cpu"

    parameter = torch.nn.Parameter(torch.tensor([10.0 + rank]))
    parameter.grad = torch.tensor([1.0 + 2.0 * rank])
    distributed.average_gradients([parameter])

    norm = RunningNorm(1, eps=0.0)
    norm.update(torch.tensor([[2.0 * rank], [2.0 * rank + 4.0]]))

    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(7.0 + rank)
    distributed.broadcast_module(module)

    result = {
        "rank": distributed.rank(),
        "world_size": distributed.world_size(),
        "grad": float(parameter.grad),
        "mean": float(norm.mean),
        "var": float(norm.var),
        "count": float(norm.count),
        "weight": float(module.weight),
        "metric_mean": distributed.mean_scalar(float(rank), "cpu"),
        "pair": distributed.sum_pair(rank + 1, 2 * rank + 1, "cpu"),
    }
    Path(output, f"rank_{rank}.json").write_text(json.dumps(result))
    distributed.shutdown()


def test_two_rank_gradient_normalizer_and_state_sync(tmp_path: Path) -> None:
    world_size = 2
    mp.spawn(_worker, args=(world_size, _free_port(), str(tmp_path)), nprocs=world_size, join=True)
    results = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
    for rank, result in enumerate(results):
        assert result["rank"] == rank
        assert result["world_size"] == world_size
        assert result["grad"] == 2.0
        assert result["mean"] == 3.0
        assert result["var"] == 5.0
        assert result["count"] == 4.0
        assert result["weight"] == 7.0
        assert result["metric_mean"] == 0.5
        assert result["pair"] == [3.0, 4.0]
