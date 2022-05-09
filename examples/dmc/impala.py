import os

import hydra
import omegaconf
import tinyspace
import torch.nn as nn

from multibeast.agents import impala
from multibeast.builder import __FeatureExtractor__, __MakeEnv__

from .environment import create_env


@__MakeEnv__.register()
def make_dmc_task(env_flags):
    env = create_env(env_flags)
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    env.seed(seed)
    env = tinyspace.convert_gymenv_spaces(env)
    return env


# from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/atari/models.py#L15
@__FeatureExtractor__.register()
class FeatureExtractor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

    def forward(self, x):
        return


# Override config_path via --config_path.
@hydra.main(config_path=".", config_name="config")
def main(cfg: omegaconf.DictConfig):
    if cfg.env.from_pixels:

        from ..atari.impala import ResNetEncoder

        try:
            __FeatureExtractor__.register(ResNetEncoder)
        except AssertionError:  # already registered
            pass
        feature_extractor = dict(cls=ResNetEncoder.__name__)

        # feature_extractor = dict(cls="FeatureExtractor")
    else:
        feature_extractor = None

    # action_dist_params = dict(cls="DiscretizedLogisticMixture", num_bins=256)
    # policy_params = dict(cls="MixturePolicyNet", n_mixtures=5, const_var=True, hidden_dim=None)

    action_dist_params = None
    policy_params = dict(
        cls="PolicyNet",
        learn_std=False,
    )

    new_flags = dict(
        use_dist_entropy=True,
        feature_extractor=feature_extractor,
        policy_params=policy_params,
        action_dist_params=action_dist_params,
        use_moolib_envpool=False,
        make_env_name="make_dmc_task",
        num_actor_processes=32,
        tags="",
        notes="",
    )
    new_flags = omegaconf.OmegaConf.create(new_flags)
    cfg = omegaconf.OmegaConf.merge(new_flags, cfg)

    impala.run(cfg)


if __name__ == "__main__":
    main()
