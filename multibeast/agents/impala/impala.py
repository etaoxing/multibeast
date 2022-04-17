import dataclasses
import logging
from typing import Optional

import torch
import torch.nn as nn
from moolib.examples import common

from multibeast.builder import __Agent__, __FeatureExtractor__

from . import vtrace
from .impala_net import ImpalaNet


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
    def _create_optimizer(FLAGS, model):
        return torch.optim.Adam(
            model.parameters(),
            lr=FLAGS.optimizer.learning_rate,
            betas=(FLAGS.optimizer.beta_1, FLAGS.optimizer.beta_2),
            eps=FLAGS.optimizer.epsilon,
        )

    @staticmethod
    def _create_scheduler(FLAGS, optimizer):
        factor = FLAGS.unroll_length * FLAGS.virtual_batch_size / FLAGS.total_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(1 - epoch * factor, 0))
        return scheduler

    @staticmethod
    def create_agent(FLAGS, observation_space, action_space):
        model = Impala._create_model(FLAGS, observation_space, action_space)
        optimizer = Impala._create_optimizer(FLAGS, model)
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
        return vtrace.compute_gradients(FLAGS, data, learner_state, stats)
