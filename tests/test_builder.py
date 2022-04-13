import pytest
import torch

from multibeast.builder import __Distribution__, __FeatureExtractor__, __MakeEnv__, __PolicyNet__
from multibeast.utils import get_action_params

from .mock_env import MockEnv


def test_MakeEnv_build():
    @__MakeEnv__.register()
    def make_env(env_flags):
        return MockEnv(**env_flags)

    action_space_cls = "discrete"
    create_env_fn = __MakeEnv__.build("make_env", dict(action_space_cls=action_space_cls))
    env = create_env_fn()
    assert env.action_space["cls"] == action_space_cls

    assert __MakeEnv__._undo_register(make_env.__name__)


def test_FeatureExtractor_build():
    @__FeatureExtractor__.register()
    class MockModule:  # noqa: B903
        def __init__(self, observation_space, action_space, n_layers=3):
            self.n_layers = n_layers

    env = MockEnv(action_space_cls="box")

    feature_extractor_flags = dict(cls="MockModule", n_layers=5)
    feature_extractor = __FeatureExtractor__.build(feature_extractor_flags, env.observation_space, env.action_space)
    assert feature_extractor.n_layers == 5

    assert __FeatureExtractor__._undo_register(MockModule.__name__)


def test_PolicyNet_build():
    @__PolicyNet__.register()
    class MockModule:  # noqa: B903
        def __init__(self, feature_dim, action_dim, action_dist_params, n_layers=3):
            self.n_layers = n_layers

            self.action_dist_cls = __Distribution__.build(action_dist_params)

        def action_dist(self, policy_logits):
            return self.action_dist_cls(logits=policy_logits)

    env = MockEnv(action_space_cls="discrete")

    action_dist_params = None
    num_actions, max_action, action_dist_params = get_action_params(env.action_space, action_dist_params)

    policy_net_flags = dict(cls="MockModule", n_layers=5)
    policy_net = __PolicyNet__.build(policy_net_flags, 128, num_actions, action_dist_params)
    assert policy_net.n_layers == 5
    print(policy_net.action_dist_cls)
    policy_net.action_dist(torch.randn((20, 4)))

    assert __PolicyNet__._undo_register(MockModule.__name__)
