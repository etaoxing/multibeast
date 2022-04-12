import numpy as np


class MockEnv:
    def __init__(self, action_space_cls):
        self._observation_space = dict(
            shape=(3, 64, 64),
            low=0,
            high=255,
            dtype=np.uint8,
        )

        if action_space_cls == "discrete":
            action_space = dict(
                cls="discrete",
                shape=(1,),
                low=0,
                high=10,
                dtype=np.int64,
            )
        elif action_space_cls == "box":
            action_space = dict(
                cls="box",
                shape=(10,),
                low=-1.0,
                high=1.0,
                dtype=np.int64,
            )
        else:
            raise ValueError

        self._action_space = action_space

    @property
    def info_keys_custom(self):
        return ["progress", "success", "custom_metric"]

    @property
    def max_episode_steps(self):
        return 20

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        pass

    def step(self, action):
        pass
