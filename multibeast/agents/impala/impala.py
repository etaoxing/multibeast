import dataclasses
import logging
from typing import Optional

import torch
import torch.nn as nn
from moolib.examples import common
from moolib.examples.common import nest

from multibeast.builder import __Agent__, __FeatureExtractor__, __Optimizer__

from .impala_net import ImpalaNet
from .vtrace import compute_vtrace


@dataclasses.dataclass
class ImpalaLearnerState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    model_version: int = 0
    num_previous_leaders: int = 0
    train_time: float = 0
    last_checkpoint: float = 0
    last_checkpoint_history: float = 0
    global_stats: Optional[dict] = None

    def save(self):
        r = dataclasses.asdict(self)
        r["model"] = self.model.state_dict()
        r["optimizer"] = self.optimizer.state_dict()
        r["scheduler"] = self.scheduler.state_dict()
        return r

    def load(self, state):
        for k, v in state.items():
            if k not in ("model", "optimizer", "scheduler", "global_stats"):
                setattr(self, k, v)
        self.model_version = state["model_version"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

        for k, v in state["global_stats"].items():
            if k in self.global_stats:
                self.global_stats[k] = type(self.global_stats[k])(**v)


@__Agent__.register()
class Impala:
    def __new__(cls):
        raise TypeError("Static class cannot be instantiated")

    @staticmethod
    def _create_model(FLAGS, observation_space, action_space):
        if FLAGS.feature_extractor:
            feature_extractor = __FeatureExtractor__.build(
                FLAGS.feature_extractor,
                observation_space,
                action_space,
            )
        else:
            feature_extractor = None

        action_dist_params = FLAGS.get("action_dist_params", None)
        policy_params = FLAGS.get("policy_params", dict(cls="PolicyNet"))

        model = ImpalaNet(
            observation_space,
            action_space,
            feature_extractor=feature_extractor,
            action_dist_params=action_dist_params,
            policy_params=policy_params,
            use_lstm=FLAGS.use_lstm,
        )

        logging.info(f"model: \n{model}")

        return model

    @staticmethod
    def _create_scheduler(FLAGS, optimizer):
        factor = FLAGS.unroll_length * FLAGS.virtual_batch_size / FLAGS.total_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(1 - epoch * factor, 0))
        return scheduler

    @staticmethod
    def create_agent(FLAGS, observation_space, action_space):
        model = Impala._create_model(FLAGS, observation_space, action_space)
        optimizer = __Optimizer__.build(FLAGS.optimizer, model.parameters())
        scheduler = Impala._create_scheduler(FLAGS, optimizer)
        learner_state = ImpalaLearnerState(model, optimizer, scheduler)
        return model, learner_state

    @staticmethod
    def create_stats(FLAGS):
        stats_keys = [
            "entropy_loss",
            "pg_loss",
            "baseline_loss",
            "kl_loss",
            "total_loss",
            "unclipped_grad_norm",
        ]

        stats = {}
        for k in stats_keys:
            stats[k] = common.StatMean()
        return stats

    @staticmethod
    def step_optimizer(FLAGS, learner_state, stats):
        unclipped_grad_norm = nn.utils.clip_grad_norm_(learner_state.model.parameters(), FLAGS.grad_norm_clipping)
        learner_state.optimizer.step()
        learner_state.scheduler.step()
        learner_state.model_version += 1

        stats["unclipped_grad_norm"] += unclipped_grad_norm.item()
        stats["optimizer_steps"] += 1
        stats["model_version"] += 1
        return

    @staticmethod
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

        # an attempt at supporting continuous action spaces based on
        # https://github.com/google-research/seed_rl/search?q=continuous&type=commits

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

        # TODO target entropy adjustment: https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/agents/vtrace/learner.py#L127
        entropy_loss = FLAGS.entropy_cost * -target_policy_action_dist.entropy().mean()

        log_likelihoods = target_action_log_probs  # target_policy_action_dist.log_prob(actions)
        pg_loss = -torch.mean(log_likelihoods * vtrace_returns.pg_advantages.detach())  # policy gradient

        baseline_advantages = vtrace_returns.vs - learner_outputs["baseline"]
        baseline_loss = FLAGS.baseline_cost * (0.5 * torch.mean(baseline_advantages**2))

        # from https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/agents/vtrace/learner.py#L123
        # KL(old_policy|new_policy) loss
        kl = behavior_action_log_probs - target_action_log_probs
        kl_loss = FLAGS.kl_cost * torch.mean(kl)
        # TODO: could also use `D.kl_divergence(behavior_policy_action_dist, target_policy_action_dist)`

        total_loss = entropy_loss + pg_loss + baseline_loss + kl_loss
        total_loss.backward()

        stats["entropy_loss"] += entropy_loss.item()
        stats["pg_loss"] += pg_loss.item()
        stats["baseline_loss"] += baseline_loss.item()
        stats["kl_loss"] += kl_loss.item()
        stats["total_loss"] += total_loss.item()
        return
