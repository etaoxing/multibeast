import dataclasses
from typing import Dict

import torch
import torch.nn as nn
from tinyspace import Space, sample_from_space

from multibeast.builder import __PolicyNet__
from multibeast.distributions import get_action_params


@dataclasses.dataclass
class ExampleLearnerState:
    r"""See impala.ImpalaLearnerState"""
    pass


class ExampleModel(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_dist_params: Dict = None,
        policy_params: Dict = None,
        feature_extractor: nn.Module = None,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.num_actions, self.max_action, self.action_dist_params = get_action_params(
            action_space,
            action_dist_params,
        )

        fc_out_features = 256

        self.feature_extractor = feature_extractor
        if feature_extractor:
            with torch.no_grad():
                x = sample_from_space(self.observation_space, batch_size=(1, 1), to_torch_tensor=True)
                feature_dim = tuple(self.feature_extractor(x).squeeze(0).shape)[0]
            self.fc = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(feature_dim, fc_out_features), nn.ReLU(inplace=True))
        else:
            feature_dim = observation_space["shape"][0]
            self.fc = nn.Sequential(nn.Linear(feature_dim, fc_out_features), nn.ReLU(inplace=True))

        core_output_size = fc_out_features

        self.policy = __PolicyNet__.build(policy_params, core_output_size, self.num_actions, self.action_dist_params)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, inputs, core_state=None, deterministic: bool = False):
        prev_action = inputs["prev_action"]
        reward = inputs["reward"]
        x = inputs["state"]
        T, B = reward.shape

        # see impala.ImpalaNet for example on using core_state for lstm

        output = dict()
        return output, core_state


class ExampleAgent:
    @staticmethod
    def create_agent(FLAGS, observation_space, action_space):
        r"""This function creates a `nn.Module` model and `LearnerState` for optimizing an agent.
        returns: `nn.Module`, `LearnerState`
        """
        raise NotImplementedError

    @staticmethod
    def step_optimizer(FLAGS, learner_state, stats):
        r"""This function should call `optimizer.step()` to update model parameters.
        returns: `None`
        """
        raise NotImplementedError

    @staticmethod
    def compute_gradients(FLAGS, data, learner_state, stats):
        r"""This function should compute a loss and call `loss.backward()`.
        returns: `None`
        """
        raise NotImplementedError

    @staticmethod
    def create_stats(FLAGS):
        r"""This function return a list of strings representing keys of statistics to track.
        returns: `List[str]`
        """
        raise NotImplementedError
