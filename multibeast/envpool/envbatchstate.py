# modified from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/common/__init__.py#L154

import moolib
import torch
from moolib.examples.common import RunningMeanStd


class EnvBatchState:
    r"""
    Also tracks ["success", "progress"] at the end of the episode, if those keys are in the info dict.

    Args:
        flags:
        model:
        zero_action: a tensor of zeros with shape [`flags.actor_batch_size`, `env.action_space.action_dim`]
        info_custom_keys: a list of keys to track stats for from the info dict.
    """

    def __init__(self, flags, model, zero_action, with_scaled_reward: bool = False, info_keys_custom: list = None):
        batch_size = flags.actor_batch_size
        device = flags.device
        self.batch_size = batch_size
        self.prev_action = zero_action.clone().to(device)
        self.future = None
        self.core_state = model.initial_state(batch_size=batch_size)
        self.core_state = tuple(s.to(device) for s in self.core_state)
        self.initial_core_state = self.core_state

        self.running_reward = torch.zeros(batch_size)
        self.step_count = torch.zeros(batch_size)

        self.with_scaled_reward = with_scaled_reward
        if with_scaled_reward:
            self.discounting = flags.discounting
            self.weighted_returns = torch.zeros(batch_size)
            self.weighted_returns_rms = RunningMeanStd()

        self.info_keys_custom = info_keys_custom

        self.time_batcher = moolib.Batcher(flags.unroll_length + 1, device)

    def update(self, env_outputs, action, stats):
        self.prev_action = action
        self.running_reward += env_outputs["reward"]

        if self.with_scaled_reward:
            self.weighted_returns *= self.discounting
            self.weighted_returns += env_outputs["reward"]
            self.weighted_returns_rms.update(self.weighted_returns)

            self.scaled_reward = env_outputs["reward"] / torch.sqrt(self.weighted_returns_rms.var + 1e-8)

        self.step_count += 1

        info = env_outputs.get("info", {})

        # TODO: https://arxiv.org/abs/1712.00378
        # timeout = info["timeout"]
        done = env_outputs["done"]

        episode_return = self.running_reward * done
        episode_step = self.step_count * done

        episodes_done = done.sum().item()
        if episodes_done > 0:
            stats["mean_episode_return"] += episode_return.sum().item() / episodes_done
            stats["mean_episode_step"] += episode_step.sum().item() / episodes_done
            if "success" in info.keys():
                stats["end_episode_success"] += (info["success"] * done).sum().item() / episodes_done
            if "progress" in info.keys():
                stats["end_episode_progress"] += (info["progress"] * done).sum().item() / episodes_done

            if self.info_keys_custom:
                for k in self.info_keys_custom:
                    stats[f"end_{k}"] += (info[k] * done).sum().item() / episodes_done

        stats["steps_done"] += done.numel()
        stats["episodes_done"] += episodes_done

        stats["running_reward"] += self.running_reward.mean().item()
        stats["running_step"] += self.step_count.mean().item()

        not_done = ~done

        self.running_reward *= not_done
        self.step_count *= not_done
        if self.with_scaled_reward:
            self.weighted_returns *= not_done
