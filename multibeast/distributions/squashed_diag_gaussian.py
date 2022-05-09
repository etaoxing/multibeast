import torch
import torch.distributions as D

from ..builder import __Distribution__


class TanhBijector(object):
    """Bijective transformation of a probability distribution using a squashing function (tanh).

    TODO: use Pyro instead (https://pyro.ai/)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """Inverse of Tanh.

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))


@__Distribution__.register()
class SquashedDiagGaussian(D.Distribution):
    r"""Also called a TanhNormal distribution.

    from https://github.com/DLR-RM/stable-baselines3/blob/ed308a71be24036744b5ad4af61b083e4fbdf83c/stable_baselines3/common/distributions.py#L195

    Also see:
    - https://github.com/rlworkgroup/garage/blob/c56513f42be9cba2ef5426425a8ad36097e679c2/src/garage/torch/distributions/tanh_normal.py
    - https://github.com/ikostrikov/jaxrl/blob/5e044e1fd080223ef5e71848b9d0f72597f2ab2f/jax_rl/networks/policies.py#L15
    """

    arg_constraints = D.Normal.arg_constraints
    support = D.Normal.support
    has_rsample = True

    def __init__(self, loc, scale, eps: float = 1e-6, validate_args=None):
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[len(batch_shape) + 1 :]
        self.loc = loc
        self.scale = scale
        super().__init__(batch_shape, event_shape, validate_args)
        self.gaussian = D.Independent(D.Normal(loc, scale), 1)
        # Avoid NaN (prevents division by zero or log of zero)
        self.eps = eps

    def _squash_correction(self, value):
        # see https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/policies/gaussian_policy.py#L131
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        return torch.sum(torch.log(1 - value**2 + self.eps), dim=1)

    def log_prob(self, value):
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability

        # It will be clipped to avoid NaN when inversing tanh
        gaussian_value = TanhBijector.inverse(value)

        # Log likelihood for a Gaussian distribution
        log_prob = self.gaussian.log_prob(gaussian_value)
        # don't need https://github.com/DLR-RM/stable-baselines3/blob/ed308a71be24036744b5ad4af61b083e4fbdf83c/stable_baselines3/common/distributions.py#L100
        # since we use D.Independent

        if len(value.shape) == 3:  # deal with (T, B, ...) shaped inputs
            T, B = log_prob.shape
            log_prob = log_prob.view(T * B) - self._squash_correction(value.view(T * B, -1))
            log_prob = log_prob.view(T, B)
        else:
            log_prob -= self._squash_correction(value)
        return log_prob

    def rsample(self, sample_shape=torch.Size()):
        sampled = self.gaussian.rsample(sample_shape=sample_shape)
        return torch.tanh(sampled)

    def mode(self):
        r"""Deterministically returns the most likely sample from the distribution."""
        return torch.tanh(self.gaussian.mean)

    def entropy(self):
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        # return None

        # just return the underlying gaussian's entropy
        return self.gaussian.entropy()
