import os

import hydra
import omegaconf
import tinyspace

# import impala here, otherwise will get:
# "omegaconf.errors.UnsupportedInterpolationType: Unsupported interpolation type uid"
from multibeast.agents import impala
from multibeast.builder import __FeatureExtractor__, __MakeEnv__

from .environment import create_env
from .resnet import ResNetEncoder


@__MakeEnv__.register()
def make_env_atari(env_flags):
    env = create_env(env_flags)
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    base_env = env.env.environment
    base_env.seed(seed)
    # print(base_env, seed)
    env = tinyspace.convert_gymenv_spaces(env)
    return env


__FeatureExtractor__.register(ResNetEncoder)


# Override config_path via --config_path.
@hydra.main(config_path=".", config_name="config")
def main(cfg: omegaconf.DictConfig):
    new_flags = dict(
        feature_extractor=dict(cls=ResNetEncoder.__name__),
        use_moolib_envpool=True,
        make_env_name="make_env_atari",
    )
    new_flags = omegaconf.OmegaConf.create(new_flags)
    cfg = omegaconf.OmegaConf.merge(new_flags, cfg)

    impala.run(cfg)


if __name__ == "__main__":
    main()
