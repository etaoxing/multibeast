import pytest

from multibeast.builder import FeatureExtractorRegistry, MakeEnvRegistry

from .mock_env import MockEnv


def test_MakeEnvRegistry():
    @MakeEnvRegistry.register()
    def make_env(**kwargs):
        return MockEnv(**kwargs)

    action_space_cls = "discrete"
    create_env_fn = MakeEnvRegistry.build("make_env", dict(action_space_cls=action_space_cls))
    env = create_env_fn()
    assert env.action_space["cls"] == action_space_cls

    assert MakeEnvRegistry._undo_register(make_env.__name__)


def test_FeatureExtractorRegistry():
    @FeatureExtractorRegistry.register()
    class MockModule:  # noqa: B903
        def __init__(self, observation_space, action_space, n_layers=3):
            self.n_layers = n_layers

    env = MockEnv(action_space_cls="box")

    feature_extractor_flags = dict(cls="MockModule", n_layers=5)
    feature_extractor = FeatureExtractorRegistry.build(env.observation_space, env.action_space, feature_extractor_flags)
    assert feature_extractor.n_layers == 5

    assert FeatureExtractorRegistry._undo_register(MockModule.__name__)
