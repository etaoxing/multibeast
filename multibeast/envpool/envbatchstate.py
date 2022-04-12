# modified from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/common/__init__.py#L154

import moolib
import torch
from moolib.examples.common import RunningMeanStd


class EnvBatchState:
    def __init__(self, flags, model):
        batch_size = flags.actor_batch_size
        device = flags.device
        self.batch_size = batch_size
        self.prev_action = torch.zeros(batch_size).long().to(device)
        self.future = None
        self.core_state = model.initial_state(batch_size=batch_size)
        self.core_state = tuple(s.to(device) for s in self.core_state)
        self.initial_core_state = self.core_state

        self.running_reward = torch.zeros(batch_size)
        self.step_count = torch.zeros(batch_size)

        self.discounting = flags.discounting
        self.weighted_returns = torch.zeros(batch_size)
        self.weighted_returns_rms = RunningMeanStd()

        self.time_batcher = moolib.Batcher(flags.unroll_length + 1, device)

    def update(self, env_outputs, action, stats):
        self.prev_action = action
        self.running_reward += env_outputs["reward"]
        self.weighted_returns *= self.discounting
        self.weighted_returns += env_outputs["reward"]
        self.weighted_returns_rms.update(self.weighted_returns)

        self.scaled_reward = env_outputs["reward"] / torch.sqrt(self.weighted_returns_rms.var + 1e-8)

        self.step_count += 1

        done = env_outputs["done"]

        episode_return = self.running_reward * done
        episode_step = self.step_count * done

        episodes_done = done.sum().item()
        if episodes_done > 0:
            stats["mean_episode_return"] += episode_return.sum().item() / episodes_done
            stats["mean_episode_step"] += episode_step.sum().item() / episodes_done
        stats["steps_done"] += done.numel()
        stats["episodes_done"] += episodes_done

        stats["running_reward"] += self.running_reward.mean().item()
        stats["running_step"] += self.step_count.mean().item()

        not_done = ~done

        self.running_reward *= not_done
        self.weighted_returns *= not_done
        self.step_count *= not_done
