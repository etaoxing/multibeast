import dataclasses


@dataclasses.dataclass
class ExampleLearnerState:
    r"""See impala.ImpalaLearnerState"""
    pass


class ExampleAgent:
    @staticmethod
    def create_agent(FLAGS, observation_space, action_space):
        r"""This function creates a `nn.Module` model and `LearnerState` for optimizing an agent.
        returns: `nn.Module`, `LearnerState`
        """
        raise NotImplementedError

    @staticmethod
    def step_optimizer(FLAGS, learner_state, stats):
        r"""This function should call `optimizer.step()` to update model parameters.
        returns: `None`
        """
        raise NotImplementedError

    @staticmethod
    def compute_gradients(FLAGS, data, learner_state, stats):
        r"""This function should compute a loss and call `loss.backward()`.
        returns: `None`
        """
        raise NotImplementedError

    @staticmethod
    def create_stats(FLAGS):
        r"""This function return a list of strings representing keys of statistics to track.
        returns: `List[str]`
        """
        raise NotImplementedError
