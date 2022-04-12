import asyncio
import multiprocessing as mp
from typing import Any

import cloudpickle
import torch
from tinyspace import collate_obs


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    try:
        while True:
            try:
                cmd, data = remote.recv()
                if cmd == "step":
                    observation, reward, done, info = env.step(data)
                    if done:
                        # # NOTE: EnvBatchState.time_batcher deals with this instead
                        # # save final observation where user can get it, then reset
                        # info["terminal_observation"] = observation
                        observation = env.reset()
                    remote.send((observation, reward, done, info))
                elif cmd == "seed":
                    remote.send(env.seed(data))
                elif cmd == "reset":
                    observation = env.reset()
                    remote.send((observation, 0.0, False, {}))
                elif cmd == "render":
                    remote.send(env.render(data))
                elif cmd == "close":
                    env.close()
                    remote.close()
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                elif cmd == "env_method":
                    method = getattr(env, data[0])
                    remote.send(method(*data[1], **data[2]))
                elif cmd == "get_attr":
                    remote.send(getattr(env, data))
                elif cmd == "set_attr":
                    remote.send(setattr(env, data[0], data[1]))
                else:
                    remote.close()
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
            except EOFError:
                break
    except KeyboardInterrupt:
        remote.close()


class EnvStepperFuture:
    def __init__(self, remotes, observation_space, callback_on_result=None):
        self.remotes = remotes
        self.observation_space = observation_space
        self.callback_on_result = callback_on_result

    def result(self):
        remotes_batch = self.remotes
        results = [remote.recv() for remote in remotes_batch]
        obses, rews, dones, infos = zip(*results)

        env_outputs = {}
        env_outputs["state"] = collate_obs(obses, self.observation_space, to_torch_tensor=True)
        env_outputs["reward"] = torch.tensor(rews, dtype=torch.float)
        env_outputs["done"] = torch.tensor(dones, dtype=torch.bool)
        env_outputs["info"] = {k: torch.tensor([x[k] for x in infos]) for k in infos[0].keys()}

        if self.callback_on_result is not None:
            self.callback_on_result()

        return env_outputs


class EnvStepper:
    r"""This class is based on stable_baselines3.SubprocVecEnv, and supports running a pool of
    unbatched CPU-based environments.

    Uses multiprocessing.Pipe instead of multiprocessing.SharedMemory, to support environments
    with observation spaces that contain variable-sized inputs (ie. point clouds).

    Args:
        env_init_fn: Callable to create a single environment.
        num_processes: Number of environments to run in parallel.
    """

    def __init__(self, env_init_fn, num_processes, start_method="spawn"):
        self.num_processes = num_processes
        self.busy_buffer = set()
        self.reset_once = set()
        self.closed = False

        env_fns = [env_init_fn] * num_processes

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_processes)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, batch_index: int, actions: torch.Tensor) -> "asyncio.Future":
        batch_size = actions.size(0)
        idx = int(batch_index * batch_size)
        remotes_batch = self.remotes[idx : idx + batch_size]

        if batch_index in self.busy_buffer:
            raise RuntimeError(f"EnvStepper: attempt to step buffer index {batch_index} twice concurrently")

        self.busy_buffer.add(batch_index)

        if batch_index not in self.reset_once:
            for remote in self.remotes:
                remote.send(("reset", None))

            self.reset_once.add(batch_index)
        else:
            actions = actions.detach().cpu().numpy()
            for remote, action in zip(remotes_batch, actions):
                remote.send(("step", action))

        def callback_on_result():
            self.busy_buffer.remove(batch_index)

        return EnvStepperFuture(remotes_batch, self.observation_space, callback_on_result)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True
