import pytest

from multibeast.builder import __FeatureExtractor__, __MakeEnv__

from .mock_env import MockEnv


def test_MakeEnv_build():
    @__MakeEnv__.register()
    def make_env(**kwargs):
        return MockEnv(**kwargs)

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
