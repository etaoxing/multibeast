# modified from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/atari/models.py

# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from moolib.examples.common import nest
from tinyspace import Space, sample_from_space

from multibeast.builder import __PolicyNet__
from multibeast.utils import get_action_params


class ImpalaNet(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_dist_params: dict = None,
        policy_params: Dict = dict(cls="PolicyNet"),  # noqa: B006
        feature_extractor: nn.Module = None,
        use_lstm: bool = False,
        fc_out_features: int = 256,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.num_actions, self.max_action, self.action_dist_params = get_action_params(
            action_space,
            action_dist_params,
        )

        self.feature_extractor = feature_extractor
        self.use_lstm = use_lstm

        if feature_extractor:
            with torch.no_grad():
                x = sample_from_space(self.observation_space, batch_size=(1, 1), to_torch_tensor=True)
                feature_dim = tuple(self.feature_extractor(x).squeeze(0).shape)[0]
            self.fc = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(feature_dim, fc_out_features), nn.ReLU(inplace=True))
        else:
            feature_dim = observation_space["shape"][0]
            self.fc = nn.Sequential(nn.Linear(feature_dim, fc_out_features), nn.ReLU(inplace=True))

        # FC output size + last action (one-hot if discrete) + last reward.
        core_output_size = fc_out_features + self.num_actions + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1)
            core_output_size = 256

        self.policy = __PolicyNet__.build(policy_params, core_output_size, self.num_actions, self.action_dist_params)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if self.use_lstm:
            return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))
        return tuple()

    def forward(self, inputs, core_state=None, deterministic: bool = False):
        prev_action = inputs["prev_action"]
        reward = inputs["reward"]
        x = inputs["state"]

        T, B = reward.shape
        if self.feature_extractor:
            x = self.feature_extractor(x)
            if len(x.shape) != 2:
                raise RuntimeError(f"Expected feature extractor to output batch of 1d features, got {x.shape}")

        x = x.view(T * B, -1)
        x = self.fc(x)

        clipped_reward = torch.clamp(reward, -1, 1).view(T * B, 1)
        if self.action_space["cls"] == "discrete":
            if prev_action.dtype != torch.long:
                # if getting "RuntimeError: one_hot is only applicable to index tensor."
                # then prev_action is not a long
                raise RuntimeError
            last_action = F.one_hot(prev_action.view(T * B), self.num_actions).float()
        elif self.action_space["cls"] == "box":
            last_action = prev_action.view(T * B, -1)
        else:
            raise ValueError
        core_input = torch.cat([x, clipped_reward, last_action], dim=-1)

        if self.use_lstm:
            done = inputs["done"]
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                core_output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(core_output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        core_output = core_output.view(T, B, -1)
        output = self.policy(core_output, deterministic=deterministic)
        if self.action_space["cls"] == "discrete":
            output["action"] = output["action"].view(T, B)

        baseline = self.baseline(core_output)
        baseline = baseline.view(T, B)
        output.update(
            baseline=baseline,
        )
        return output, core_state
