from typing import Callable

from .utils.registry import Registry


def parse_params(params):
    assert "cls" in params.keys()
    params = params.copy()
    c = params.pop("cls")
    return c, params


def build_make_env(make_env_name, make_env_kwargs) -> Callable:
    r"""This function gets a function in the registry, which can be called to make an environment.
    Example:
        ```python
        from multibeast.builder import MakeEnvRegistry

        @MakeEnvRegistry.register()
        def make_env(**kwargs):  # kwargs come from hydra_cfg.yaml
            env = gym.make("CartPole", **kwargs)
            return env

        # where `make_env_name = "make_env" == make_env.__name__`
        ```
    """
    create_env_fn = lambda: MakeEnvRegistry.get(make_env_name)(**make_env_kwargs)
    return create_env_fn


def build_feature_extractor(observation_space, action_space, feature_extractor_kwargs):
    c, params = parse_params(feature_extractor_kwargs)
    FeatureExtractorCls = FeatureExtractorRegistry.get(c)
    return FeatureExtractorCls(observation_space, action_space, **params)


MakeEnvRegistry = Registry("MakeEnv")
MakeEnvRegistry.build = build_make_env

FeatureExtractorRegistry = Registry("FeatureExtractor")
FeatureExtractorRegistry.build = build_feature_extractor
