import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from multibeast.builder import __Distribution__, __PolicyNet__


class ExamplePolicyNet(nn.Module):
    def __init__(
        self,
        core_output_size,
        num_actions,
        action_dist_params,
    ):
        self.action_dist_cls = __Distribution__.build(action_dist_params)
        self.logits = nn.Linear(core_output_size, num_actions)

    def action_dist(self, policy_logits) -> D.Distribution:
        action_dist = self.action_dist_cls(logits=policy_logits)
        return action_dist

    def forward(self, core_output, deterministic: bool = False):
        T, B, _ = core_output.shape
        x = core_output.view(T * B, -1)
        policy_logits = self.logits(x)
        action_dist = self.action_dist(policy_logits)
        action = action_dist.rsample()  # preferred if has rsample(..) to pass gradients w/ reparameterization trick
        action = action.view(T, B, -1)
        policy_logits = policy_logits.view(T, B, -1)
        return policy_logits, action


@__PolicyNet__.register()
class PolicyNet(nn.Module):
    r"""This module supports up to two outputs (loc and scale) for some distribution parameterizing the policy.

    Refs:
    - https://github.com/ikostrikov/jaxrl/blob/8ac614b0c5202acb7bb62cdb1b082b00f257b08c/jaxrl/networks/policies.py#L14
    - https://github.com/DLR-RM/stable-baselines3/blob/ed308a71be24036744b5ad4af61b083e4fbdf83c/stable_baselines3/common/distributions.py#L115

    Args:
        learn_std: If None, then do not output stdev. If True, then std is learnable. If False, then
            std is kept constant to some initial value.
        state_dependent_std: If True, then use an additional layer to predict std given the state features.
        log_std_init: Initial value for the log standard deviation (using log std in fact to allow negative values)
    """

    def __init__(
        self,
        core_output_size,
        num_actions,
        action_dist_params,
        learn_std: bool = None,
        state_dependent_std: bool = False,
        log_std_init: float = -2.0,  # stable_baselines3 default, jaxrl uses 0.0
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dist_cls = __Distribution__.build(action_dist_params)

        self.learn_std = learn_std
        self.state_dependent_std = state_dependent_std
        self.log_std_init = log_std_init
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.logits = nn.Linear(core_output_size, num_actions)

        if learn_std is not None:
            output_dim = num_actions

            if state_dependent_std:
                self._log_std = nn.Linear(core_output_size, output_dim)
            else:
                # -> diag covariance
                self._log_std = nn.Parameter(torch.ones(output_dim) * log_std_init, requires_grad=learn_std)

    def action_dist(self, policy_logits):
        if self.learn_std is None:
            return self.action_dist_cls(logits=policy_logits)
        else:
            assert isinstance(policy_logits, tuple)
            loc = policy_logits[0]
            scale = policy_logits[1]
            return self.action_dist_cls(loc, scale)

    def forward(self, core_output, deterministic: bool = False):
        T, B, _ = core_output.shape
        x = core_output.view(T * B, -1)
        policy_logits = self.logits(x)

        if self.learn_std is None:
            pass
        else:
            mu = policy_logits

            if self.state_dependent_std:
                scale = self._log_std(x)
            else:
                scale = self._log_std if self.training else self._log_std.detach()
                # NOTE: even though action_dist_cls may handle it, we expand scale to help moolib.Batcher
                scale = scale.view(1, -1).expand_as(mu)

            scale = torch.clamp(scale, self.log_std_min, self.log_std_max)
            scale = torch.exp(scale)
            policy_logits = (mu, scale)

        action_dist = self.action_dist(policy_logits)
        if deterministic:
            raise NotImplementedError
        else:
            # D.Categorical only defines sample()
            action = action_dist.rsample() if action_dist.has_rsample else action_dist.sample()

        if self.learn_std is None:
            action = action.view(T, B, -1)
            policy_logits = policy_logits.view(T, B, -1)
            return policy_logits, action
        else:
            action = action.view(T, B, -1)
            mu = mu.view(T, B, -1)
            scale = scale.view(T, B, -1)
            return (mu, scale), action


# from https://github.com/SudeepDasari/one_shot_transformers/blob/ecd43b0c182451b67219fdbc7d6a3cd912395f17/hem/models/inverse_module.py#L106
# They report "For most of our experiments, the model performed best when using two mixture
# components and learned constant variance parameters per action dimension", so `n_mixtures=2` and `const_var=True`.
# Also see https://github.com/ikostrikov/jaxrl/blob/8ac614b0c5202acb7bb62cdb1b082b00f257b08c/jaxrl/networks/policies.py#L47
@__PolicyNet__.register()
class MixturePolicyNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        action_dist_params,
        n_mixtures: int = 3,
        const_var: bool = False,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dist_cls = __Distribution__.build(action_dist_params)

        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures, self._dist_size = n_mixtures, torch.Size((out_dim, n_mixtures))

        if hidden_dim:
            self._l = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self._l = None
            hidden_dim = in_dim

        self._mu = nn.Linear(hidden_dim, out_dim * n_mixtures)
        self._const_var = const_var

        if const_var:
            # independent of state, still optimized
            ln_scale = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter("_ln_scale", nn.Parameter(ln_scale, requires_grad=True))
            # TODO: std per mixture?
        else:
            # state dependent
            self._ln_scale = nn.Linear(hidden_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(hidden_dim, out_dim * n_mixtures) if n_mixtures > 1 else None

    def action_dist(self, policy_logits_tuple):
        return self.action_dist_cls(*policy_logits_tuple)

    def forward(self, core_output, deterministic: bool = False):
        T, B, _ = core_output.shape

        x = core_output.view(T * B, -1)
        if self._l is not None:
            x = self._l(x)

        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
        if self._const_var:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, -1, 1)).expand_as(mu)
        else:
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))

        logit_prob = (
            self._logit_prob(x).reshape((x.shape[:-1] + self._dist_size)) if self._n_mixtures > 1 else torch.ones_like(mu)
        )

        policy_logits_tuple = (mu, ln_scale, logit_prob)

        action_dist = self.action_dist(policy_logits_tuple)
        if deterministic:
            raise NotImplementedError
        else:
            action = action_dist.rsample()

        action = action.view(T, B, -1)
        policy_logits_tuple = tuple(x.view(T, B, -1, self._n_mixtures) for x in policy_logits_tuple)

        return policy_logits_tuple, action
