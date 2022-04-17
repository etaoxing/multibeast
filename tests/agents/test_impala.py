import pytest
import tinyspace
import torch

from multibeast.agents.impala import ImpalaNet

from ..mock_env import MockEnv


def _get_input(env, T, B):
    x = tinyspace.sample_from_space(env.observation_space, batch_size=(T, B), to_torch_tensor=True)
    prev_action = tinyspace.sample_from_space(env.action_space, batch_size=(T, B), to_torch_tensor=True)
    inputs = dict(
        prev_action=prev_action,
        reward=torch.randn((T, B)),
        state=x,
    )
    return inputs


@pytest.mark.order(1)
def test_impala_forward_Categorical():
    env = MockEnv(obs_space_type="1d", action_space_cls="discrete")

    T, B = 20, 4
    inputs = _get_input(env, T=T, B=B)

    # test if default is set properly
    policy_params = dict(cls="PolicyNet")
    action_dist_params = dict(cls="Categorical")
    model = ImpalaNet(env.observation_space, env.action_space)
    model = ImpalaNet(
        env.observation_space,
        env.action_space,
        policy_params=policy_params,
        action_dist_params=action_dist_params,
    )

    outputs, core_state = model(inputs)

    assert outputs["policy_logits"].shape == (T, B, model.num_actions)
    assert outputs["action"].shape == (T, B)
    assert outputs["baseline"].shape == (T, B)

    # try creating distribution
    action_dist = model.policy.action_dist(outputs["policy_logits"])
    assert action_dist.log_prob(outputs["action"]).shape == (T, B)


@pytest.mark.order(2)
def test_impala_forward_SquashedDiagGaussian():
    env = MockEnv(obs_space_type="1d", action_space_cls="box")

    T, B = 20, 4
    inputs = _get_input(env, T=T, B=B)

    policy_params = dict(cls="PolicyNet", learn_std=False)
    action_dist_params = dict(cls="SquashedDiagGaussian")
    model = ImpalaNet(
        env.observation_space,
        env.action_space,
        policy_params=policy_params,
        action_dist_params=action_dist_params,
    )
    outputs, core_state = model(inputs)

    loc = outputs["policy_logits"][0]
    scale = outputs["policy_logits"][1]
    assert loc.shape == (T, B, model.num_actions)
    assert scale.shape == (T, B, model.num_actions)
    assert outputs["action"].shape == (T, B, model.num_actions)
    assert outputs["baseline"].shape == (T, B)

    # try creating distribution
    action_dist = model.policy.action_dist(outputs["policy_logits"])
    assert action_dist.log_prob(outputs["action"]).shape == (T, B)


@pytest.mark.order(2)
def test_impala_forward_DiscretizedLogisticMixture():
    env = MockEnv(obs_space_type="1d", action_space_cls="box")

    T, B = 20, 4
    inputs = _get_input(env, T, B)

    action_dist_params = dict(cls="DiscretizedLogisticMixture", num_bins=256)
    policy_params = dict(cls="MixturePolicyNet", n_mixtures=5, const_var=False, hidden_dim=None)
    model = ImpalaNet(
        env.observation_space,
        env.action_space,
        action_dist_params=action_dist_params,
        policy_params=policy_params,
    )
    outputs, core_state = model(inputs)

    for x in outputs["policy_logits"]:
        assert x.shape == (T, B, model.num_actions, model.policy._n_mixtures)
    assert outputs["action"].shape == (T, B, model.num_actions)
    assert outputs["baseline"].shape == (T, B)

    # try creating distribution
    action_dist = model.policy.action_dist(outputs["policy_logits"])
    assert action_dist.log_prob(outputs["action"]).shape == (T, B, model.num_actions)
