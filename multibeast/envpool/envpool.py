# based on https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/common/__init__.py#L154

import asyncio
from typing import Any, Callable

import moolib
import torch

from .envstepper import EnvStepper


class EnvPool:
    r"""The class maintains `num_batches` batches of environments, with `batch_size` environments each.
    This means the batches can be stepped through alternately, for increased efficiency
    (cf "double-buffering"), and the whole `EnvPool` uses ``num_process` to run these environments.

    We use this instead of `moolib@b785684.EnvPool` b/c moolib's version only supports environments with
    discrete action spaces. Furthermore, moolib's version uses RPC and C++ threads, while this class is
    entirely in Python for simplicity. This class supports GPU-based environments (like IsaacGym) which
    are already batched, and provides a way to batch standard Gym-like environments using multiple processes.

    Args:
        create_env (Callable[[], Env]): a user-defined function that returns a Gym-like environment.
        num_processes: how many processes should be used for running environments.
        batch_size: the number of environments in one batch.
        num_batches: the number of batches to maintain (for double-buffering).
    """

    def __init__(
        self,
        env_init_fn: Callable[[], Any],
        num_processes: int,
        batch_size: int,
        num_batches: int,
    ):
        assert num_processes / batch_size == num_batches

        self.num_processes = num_processes
        self.batch_size = batch_size
        self.num_batches = num_batches

        self.env_stepper = self.spawn(env_init_fn)

    def spawn(self, env_init_fn):
        self.name = moolib.create_uid()

        EnvStepperCls = EnvStepper  # for unbatched, CPU-based environments (ie. Gym)
        # EnvStepperCls = BatchedEnvStepper  # use as a wrapper for batched, CPU-based environments (ie. sail-sg/envpool)
        # EnvStepperCls = BatchedCUDAEnvStepper  # for batched, GPU-based environments that use CUDA tensors (ie. IsaacGym)

        env_stepper = EnvStepperCls(env_init_fn, self.num_processes)
        return env_stepper

    def step(self, batch_index: int, action: torch.Tensor) -> "asyncio.Future":
        r"""
        Step through a batch of envs.

        Args:
            batch_index: index of the batch of envs are we stepping.
            action: actions for each of the envs.
        """
        if action.size(0) != self.batch_size:
            raise RuntimeError(
                f"env step was passed an action tensor with batch size {action.size(0)}, expected {self.batch_size}"
            )

        if batch_index < 0 or batch_index >= self.num_batches:
            raise RuntimeError(
                f"env step was passed an out-of-range batch index {batch_index} (valid range is [0,{self.num_batches})"
            )

        return self.env_stepper.step(batch_index, action)
