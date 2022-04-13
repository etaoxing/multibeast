import pytest
import tinyspace
import torch

from multibeast.agents.impala import ImpalaNet

from ..mock_env import MockEnv


def _get_input(env, T=20, B=4):
    x = tinyspace.sample_from_space(env.observation_space, batch_size=(T, B), to_torch_tensor=True)
    prev_action = tinyspace.sample_from_space(env.action_space, batch_size=(T, B), to_torch_tensor=True)
    inputs = dict(
        prev_action=prev_action,
        reward=torch.ones((T, B)),
        state=x,
    )
    return inputs


@pytest.mark.order(1)
def test_impala_discrete_Categorical_forward():
    env = MockEnv(obs_space_type="1d", action_space_cls="discrete")
    inputs = _get_input(env)
    inputs["prev_action"] = inputs["prev_action"].squeeze(-1)

    # test if default is set properly
    model = ImpalaNet(env.observation_space, env.action_space)
    outputs, core_state = model(inputs)

    # test override
    action_dist_params = dict(cls="Categorical")
    model = ImpalaNet(env.observation_space, env.action_space, action_dist_params=action_dist_params)
