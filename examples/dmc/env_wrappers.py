import gym
import numpy as np


# from https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/common/env_wrappers.py#L72
class DiscretizeEnvWrapper(gym.Env):
    """Wrapper for discretizing actions."""

    def __init__(self, env, n_actions_per_dim, discretization="lin", action_ratio=None):
        """Discretize actions.

        Args:
          env: Environment to be wrapped.
          n_actions_per_dim: The number of buckets per action dimension.
          discretization: Discretization mode, can be 'lin' or 'log',
            'lin' spaces buckets linearly between low and high while 'log'
            spaces them logarithmically.
          action_ratio: The ratio of the highest and lowest positive action
            for logarithim discretization.
        """
        self.env = env
        assert len(env.action_space.shape) == 1
        dim_action = env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete([n_actions_per_dim] * dim_action)
        self.observation_space = env.observation_space
        high = env.action_space.high
        if isinstance(high, float):
            assert env.action_space.low == -high
        else:
            high = high[0]
            assert (env.action_space.high == [high] * dim_action).all()
            assert (env.action_space.low == -env.action_space.high).all()
        if discretization == "log":
            assert n_actions_per_dim % 2 == 1, (
                "The number of actions per dimension " "has to be odd for logarithmic discretization."
            )
            assert action_ratio is not None
            log_range = np.linspace(np.log(high / action_ratio), np.log(high), n_actions_per_dim // 2)
            self.action_set = np.concatenate([-np.exp(np.flip(log_range)), [0.0], np.exp(log_range)])
        elif discretization == "lin":
            self.action_set = np.linspace(-high, high, n_actions_per_dim)

    def step(self, action):
        assert self.action_space.contains(action)
        action = np.take(self.action_set, action)
        assert self.env.action_space.contains(action)
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# modified from https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/mujoco/env.py#L29
class SinglePrecisionWrapper(gym.Wrapper):
    """Single precision Wrapper for Mujoco environments."""

    def __init__(self, env):
        """Initialize the wrapper.

        Args:
          env: MujocoEnv to be wrapped.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(self.observation_space.low, self.observation_space.high, dtype=np.float32)

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(reward, np.ndarray):
            reward = reward.astype(np.float32)
        else:
            reward = float(reward)
        return obs.astype(np.float32), reward, done, info
