# Taken from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/common/vtrace.py#L156

# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F
from moolib.examples.common import nest

VTraceReturns = collections.namedtuple("VTraceReturns", ["vs", "pg_advantages"])


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    lambda_=1.0,
):
    r"""V-trace from log importance weights.

    Calculates V-trace actor critic targets as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions. This code also supports the
    case where all tensors have the same number of additional dimensions, e.g.,
    `rewards` is [T, B, C], `values` is [T, B, C], `bootstrap_value` is [B, C].

    Args:
      log_rhos: A float32 tensor of shape [T, B, NUM_ACTIONS] representing the log
        importance sampling weights, i.e.
        log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
        on rhos in log-space for numerical stability.
      discounts: A float32 tensor of shape [T, B] with discounts encountered when
        following the behaviour policy.
      rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
        importance weights (rho) when calculating the baseline targets (vs).
        rho^bar in the paper. If None, no clipping is applied.
      clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
        on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
        None, no clipping is applied.
      lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). See Remark 2
        in paper. Defaults to lambda_=1.
      name: The name scope that all V-trace operations will be created in.

    Returns:
      A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape [T, B]. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
          advantage in the calculation of policy gradients.
    """
    # from https://github.com/ray-project/ray/blob/c61910487fc01efefed6ef759d461970c7b2f974/rllib/agents/impala/vtrace_torch.py#L308
    # Make sure tensor ranks are consistent.
    rho_rank = len(log_rhos.size())  # Usually 2.
    assert rho_rank == len(values.size())
    assert rho_rank - 1 == len(bootstrap_value.size()), "must have rank {}".format(rho_rank - 1)
    assert rho_rank == len(discounts.size())
    assert rho_rank == len(rewards.size())

    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.0)
    if lambda_ is not None:
        cs *= lambda_
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    acc = torch.zeros_like(bootstrap_value)
    result = []
    for t in range(discounts.shape[0] - 1, -1, -1):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        result.append(acc)
    result.reverse()
    vs_minus_v_xs = torch.stack(result)

    # Add V(x_s) to get v_s.
    vs = torch.add(vs_minus_v_xs, values)

    # Advantage for policy gradient.
    broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
    vs_t_plus_1 = torch.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    # Make sure no gradients backpropagated through the returned values.
    return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def compute_vtrace(
    learner_outputs,
    behavior_action_log_probs,
    target_action_log_probs,
    discounts,
    rewards,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    values = learner_outputs["baseline"]
    log_rhos = target_action_log_probs - behavior_action_log_probs

    # TODO: put this on cpu? https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/experiment.py#L374
    # or move to C++ https://github.com/facebookresearch/minihack/blob/65fc16f0f321b00552ca37db8e5f850cbd369ae5/minihack/agent/polybeast/polybeast_learner.py#L342
    vtrace_returns = vtrace.from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    vtrace_returns = vtrace.VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )
    return vtrace_returns


def compute_gradients(FLAGS, data, learner_state, stats):
    model = learner_state.model

    env_outputs = data["env_outputs"]
    actor_outputs = data["actor_outputs"]
    initial_core_state = data["initial_core_state"]

    model.train()

    learner_outputs, _ = model(env_outputs, initial_core_state)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from env_outputs[t] -> action[t] to action[t] -> env_outputs[t].
    # seed_rl comment: At this point, we have unroll length + 1 steps. The last step is only used
    # as bootstrap value, so it's removed.
    learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)
    env_outputs = nest.map(lambda t: t[1:], env_outputs)
    actor_outputs = nest.map(lambda t: t[:-1], actor_outputs)

    rewards = env_outputs["reward"]
    if FLAGS.reward_clip:
        rewards = torch.clip(rewards, -FLAGS.reward_clip, FLAGS.reward_clip)

    # TODO: reward normalization ?

    discounts = (~env_outputs["done"]).float() * FLAGS.discounting

    behavior_policy_action_dist = model.policy.action_dist(actor_outputs["policy_logits"])
    target_policy_action_dist = model.policy.action_dist(learner_outputs["policy_logits"])

    actions = actor_outputs["action"]
    behavior_action_log_probs = behavior_policy_action_dist.log_prob(actions)
    target_action_log_probs = target_policy_action_dist.log_prob(actions)
    vtrace_returns = compute_vtrace(
        learner_outputs,
        behavior_action_log_probs,
        target_action_log_probs,
        discounts,
        rewards,
        bootstrap_value,
    )

    entropy_loss = FLAGS.entropy_cost * -target_policy_action_dist.entropy().mean()

    # log_likelihoods = target_policy_action_dist.log_prob(actions)
    log_likelihoods = target_action_log_probs
    pg_loss = -torch.mean(log_likelihoods * vtrace_returns.pg_advantages.detach())  # policy gradient

    baseline_advantages = vtrace_returns.vs - learner_outputs["baseline"]
    baseline_loss = FLAGS.baseline_cost * (0.5 * torch.mean(baseline_advantages**2))

    # KL(old_policy|new_policy) loss
    kl = behavior_action_log_probs - target_action_log_probs
    kl_loss = FLAGS.get("kl_cost", 0.0) * torch.mean(kl)

    # from .losses import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss
    #
    # vtrace_returns = vtrace.from_logits(
    #     behavior_policy_logits=actor_outputs["policy_logits"],
    #     target_policy_logits=learner_outputs["policy_logits"],
    #     actions=actor_outputs["action"],
    #     discounts=discounts,
    #     rewards=rewards,
    #     values=learner_outputs["baseline"],
    #     bootstrap_value=bootstrap_value,
    # )
    # entropy_loss = FLAGS.entropy_cost * compute_entropy_loss(learner_outputs["policy_logits"])
    # pg_loss = compute_policy_gradient_loss(
    #     learner_outputs["policy_logits"],
    #     actor_outputs["action"],
    #     vtrace_returns.pg_advantages,
    # )
    # baseline_loss = FLAGS.baseline_cost * compute_baseline_loss(vtrace_returns.vs - learner_outputs["baseline"])

    total_loss = entropy_loss + pg_loss + baseline_loss + kl_loss
    total_loss.backward()

    stats["env_train_steps"] += FLAGS.unroll_length * FLAGS.batch_size

    stats["entropy_loss"] += entropy_loss.item()
    stats["pg_loss"] += pg_loss.item()
    stats["baseline_loss"] += baseline_loss.item()
    stats["kl_loss"] += kl_loss.item()
    stats["total_loss"] += total_loss.item()
