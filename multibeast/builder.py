from functools import partial
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
        from multibeast.builder import __MakeEnv__

        @MakeEnvRegistry.register()
        def make_env(**kwargs):  # kwargs come from hydra_cfg.yaml
            env = gym.make("CartPole", **kwargs)
            return env

        # where `make_env_name = "make_env" == make_env.__name__`
        ```
    """
    create_env_fn = lambda: __MakeEnv__.get(make_env_name)(**make_env_kwargs)
    return create_env_fn


def build_feature_extractor(feature_extractor_params, observation_space, action_space):
    c, params = parse_params(feature_extractor_params)
    FeatureExtractorCls = __FeatureExtractor__.get(c)
    return FeatureExtractorCls(observation_space, action_space, **params)


def build_policy_net(policy_params, *args):
    c, params = parse_params(policy_params)
    PolicyNetCls = __PolicyNet__.get(c)
    return PolicyNetCls(*args, **params)


def build_distribution(distribution_params):
    r"""Returns a wrapped constructor for a torch.Distribution"""
    c, params = parse_params(distribution_params)
    try:
        # import module to load register() calls
        import multibeast.distributions  # noqa: F401

        DistributionCls = __Distribution__.get(c)
    except KeyError:
        import torch.distributions as D

        DistributionCls = getattr(D, c)
    return partial(DistributionCls, **params)


__MakeEnv__ = Registry("MakeEnv")
__MakeEnv__.build = build_make_env

__FeatureExtractor__ = Registry("FeatureExtractor")
__FeatureExtractor__.build = build_feature_extractor

__PolicyNet__ = Registry("PolicyNet")
__PolicyNet__.build = build_policy_net

__Distribution__ = Registry("Distribution")
__Distribution__.build = build_distribution
