import torch
import torch.nn.functional as F

from multibeast.builder import __Distribution__
from multibeast.distributions import get_action_params

from ..mock_env import MockEnv


def _compute_entropy_loss(logits):  # Categorical
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.mean(entropy_per_timestep)


def _compute_policy_gradient_loss(logits, actions, advantages):  # Categorical
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages.detach()
    return torch.mean(policy_gradient_loss_per_timestep)


def test_losses_categorical(test_entropy=True, test_kl=True, test_pg=True):
    env = MockEnv(obs_space_type="1d", action_space_cls="discrete")
    action_dist_params = None
    num_actions, max_action, action_dist_params = get_action_params(
        env.action_space,
        action_dist_params,
    )
    assert action_dist_params["cls"] == "Categorical"
    action_dist_cls = __Distribution__.build(action_dist_params)

    T, B = 20, 4
    learner_outputs_policy_logits = torch.randn(T, B, num_actions)

    if test_entropy:
        target_policy_action_dist = action_dist_cls(logits=learner_outputs_policy_logits)

        entropy_loss = -target_policy_action_dist.entropy().mean()
        _entropy_loss = _compute_entropy_loss(learner_outputs_policy_logits)
        assert torch.all(torch.isclose(entropy_loss, _entropy_loss))

        # --- also test flattened (T * B, -1)
        target_policy_action_dist = action_dist_cls(logits=learner_outputs_policy_logits.view(T * B, -1))
        entropy_loss = -target_policy_action_dist.entropy().mean()
        _entropy_loss = _compute_entropy_loss(learner_outputs_policy_logits.view(T * B, -1))
        assert torch.all(torch.isclose(entropy_loss, _entropy_loss))

    actions = action_dist_cls(logits=learner_outputs_policy_logits).sample()  # use sample instead of rsample for Categorical

    if test_pg:
        target_policy_action_dist = action_dist_cls(logits=learner_outputs_policy_logits)
        value_error = torch.randn((T, B))
        target_action_log_probs = target_policy_action_dist.log_prob(actions)

        pg_loss = -torch.mean(target_action_log_probs * value_error.detach())  # policy gradient
        _pg_loss = _compute_policy_gradient_loss(learner_outputs_policy_logits, actions, value_error)
        assert torch.all(torch.isclose(pg_loss, _pg_loss))

    if test_kl:
        target_policy_action_dist = action_dist_cls(logits=learner_outputs_policy_logits)

        actor_outputs_policy_logits = torch.randn(T, B, num_actions)
        behavior_policy_action_dist = action_dist_cls(logits=actor_outputs_policy_logits)

        behavior_action_log_probs = behavior_policy_action_dist.log_prob(actions)
        target_action_log_probs = target_policy_action_dist.log_prob(actions)
        kl = behavior_action_log_probs - target_action_log_probs
        kl_loss = torch.mean(kl)


def test_losses_tanhnormal(test_entropy=True):
    env = MockEnv(obs_space_type="1d", action_space_cls="box")
    action_dist_params = None
    num_actions, max_action, action_dist_params = get_action_params(
        env.action_space,
        action_dist_params,
    )
    assert action_dist_params["cls"] == "SquashedDiagGaussian"
    action_dist_cls = __Distribution__.build(action_dist_params)

    if test_entropy:
        T, B = 20, 4
        learner_outputs_policy_logits = (torch.randn(T, B, num_actions), torch.randn(T, B, num_actions).exp())
        loc, scale = learner_outputs_policy_logits

        target_policy_action_dist = action_dist_cls(loc, scale)
        entropy_loss = -target_policy_action_dist.entropy().mean()

        # --- also test flattened (T * B, -1)
        target_policy_action_dist1 = action_dist_cls(loc.view(T * B, -1), scale.view(T * B, -1))
        entropy_loss1 = -target_policy_action_dist1.entropy().mean()
        assert torch.all(torch.isclose(entropy_loss, entropy_loss1))
