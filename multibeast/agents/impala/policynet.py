import numpy as np
import torch
import torch.nn as nn

from multibeast.builder import __Distribution__, __PolicyNet__


@__PolicyNet__.register()
class PolicyNet(nn.Module):
    def __init__(
        self,
        core_output_size,
        num_actions,
        action_dist_params,
        learn_std=None,
        state_dependent_std=False,
        init_std=1.0,
        min_std=1e-6,
        max_std=None,
    ):
        super().__init__()
        self.logits = nn.Linear(core_output_size, num_actions)
        self.action_dist_cls = __Distribution__.build(action_dist_params)

        if learn_std is not None:
            output_dim = num_actions

            if state_dependent_std:
                self._log_std = nn.Linear(core_output_size, output_dim)
            else:
                self._init_std = torch.Tensor([init_std]).log()
                log_std = torch.Tensor([init_std] * output_dim).log()
                if learn_std:
                    self._log_std = torch.nn.Parameter(log_std)
                else:
                    self._log_std = log_std
                    self.register_buffer("log_std", self._log_std)

            raise NotImplementedError

    def action_dist(self, policy_logits):
        return self.action_dist_cls(logits=policy_logits)

    def forward(self, core_output, deterministic: bool = False):
        T, B, _ = core_output.shape

        policy_logits = self.logits(core_output.view(T * B, -1))

        action_dist = self.action_dist(policy_logits)
        if deterministic:
            raise NotImplementedError
        else:
            action = action_dist.sample()

        action = action.view(T, B, -1)
        policy_logits = policy_logits.view(T, B, -1)

        return dict(policy_logits=policy_logits, action=action)


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
        n_mixtures=3,
        const_var=False,
        hidden_dim=256,
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
            ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
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
            action = action_dist.sample()

        return dict(policy_logits=policy_logits_tuple, action=action)
