import torch

from multibeast.distributions import SquashedDiagGaussian
import torch.distributions as D


# from https://github.com/denisyarats/pytorch_sac/blob/7aa74312d82be95a8ae42d62d82f47a0980553da/distribs.py#L38
class SquashedNormal(D.TransformedDistribution):
    def __init__(self, loc, scale, independent=False):
        self.loc = loc
        self.scale = scale
        self.independent = independent

        self.base_dist = D.Normal(loc, scale)
        if independent:
            self.base_dist = D.Independent(self.base_dist, 1)
        transforms = [D.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def test_SquashedDiagGaussian():
    T, B = 4, 2
    num_actions = 10
    loc = torch.randn(T, B, num_actions)
    scale = torch.randn(T, B, num_actions).exp()

    n = 2

    d1 = SquashedDiagGaussian(loc, scale)
    a1 = [d1.rsample() for _ in range(n)]
    lp1 = [d1.log_prob(a) for a in a1]

    for a in a1:
        assert a.shape == (T, B, num_actions)
    for lp in lp1:
        assert lp.shape == (T, B)

    # verify that separating time & batch yields the same result as when combining them
    d1_flatten = SquashedDiagGaussian(loc.view(T * B, -1), scale.view(T * B, -1))
    for a, lp in zip(a1, lp1):
        lp_flatten = d1_flatten.log_prob(a.view(T * B, -1))
        assert lp_flatten.shape == (T * B,)
        assert torch.all(torch.isclose(lp, lp_flatten.view(T, B)))

    # # This is unstable, and will result in nans near the boundary b/c it does not clip
    # d2 = SquashedNormal(loc.view(T * B, -1), scale.view(T * B, -1), independent=True)
    # a2 = [d1.rsample() for _ in range(n)]
    # for a, lp in zip(a1, lp1):
    #     lp_flatten = d2.log_prob(a.view(T * B, -1))
    #     assert lp_flatten.shape == (T * B,)
    #     lp_ = lp_flatten.view(T, B)
    #     print(lp, lp_)
    #     assert torch.all(torch.isclose(lp, lp_))
