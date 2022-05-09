import dmc2gym
import gym

from .env_wrappers import DiscretizeEnvWrapper, SinglePrecisionWrapper


def create_env(env_flags):
    visualize_reward = env_flags.visualize_reward
    if env_flags.from_pixels:
        visualize_reward = False

    env = dmc2gym.make(
        domain_name=env_flags.domain_name,
        task_name=env_flags.task_name,
        visualize_reward=visualize_reward,
        from_pixels=env_flags.from_pixels,
        height=env_flags.height,
        width=env_flags.width,
        frame_skip=env_flags.frame_skip,
    )
    if env_flags.from_pixels:
        env = gym.wrappers.FrameStack(env, num_stack=env_flags.frame_stack)
    else:
        env = SinglePrecisionWrapper(env)

    # dmc2gym already does action space normalization to [-1, 1]
    if env_flags.get("discretization", False):
        # from https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/mujoco/env.py#L107

        n_actions_per_dim = 11
        action_ratio = 30.0
        env = DiscretizeEnvWrapper(env, n_actions_per_dim, env_flags.discretization, action_ratio)

    return env
