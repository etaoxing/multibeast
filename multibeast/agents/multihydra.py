# modified from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/vtrace/experiment.py
# and https://github.com/facebookresearch/minihack/blob/c17084885833cbeee8bdd6684d0cde0a2536e3bb/minihack/agent/polybeast/polyhydra.py

# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import dataclasses
import getpass
import logging
import os
import pprint
import signal
import socket
import time
from typing import Optional

import coolname
import hydra
import moolib
import omegaconf
import torch
import torch.nn as nn
from moolib.examples import common
from moolib.examples.common import nest, record
from tinyspace import sample_from_space

from multibeast.builder import __FeatureExtractor__, __MakeEnv__
from multibeast.envpool import EnvBatchState, EnvPool

from .impalanet import ImpalaNet
from ..record_utils import calculate_sps, load_checkpoint, log, save_checkpoint
from .vtrace import compute_gradients


@dataclasses.dataclass
class LearnerState:
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


def create_model(observation_space, action_space):
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


def create_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.optimizer.learning_rate,
        betas=(FLAGS.optimizer.beta_1, FLAGS.optimizer.beta_2),
        eps=FLAGS.optimizer.epsilon,
    )


def create_scheduler(optimizer):
    factor = FLAGS.unroll_length * FLAGS.virtual_batch_size / FLAGS.total_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(1 - epoch * factor, 0))
    return scheduler


def step_optimizer(learner_state, stats):
    unclipped_grad_norm = nn.utils.clip_grad_norm_(learner_state.model.parameters(), FLAGS.grad_norm_clipping)
    learner_state.optimizer.step()
    learner_state.scheduler.step()
    learner_state.model_version += 1

    stats["unclipped_grad_norm"] += unclipped_grad_norm.item()
    stats["optimizer_steps"] += 1
    stats["model_version"] += 1


SLUG = None
# need to cache this, otherwise uid will differ b/w what shows up in logger and hydra config
# when calling impala.run() from a different file with @hydra.main() decorator.
def uid():
    global SLUG
    if SLUG is None:
        SLUG = coolname.generate_slug(2)
    return "%s:%i:%s" % (socket.gethostname(), os.getpid(), SLUG)


omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)


def run(cfg: omegaconf.DictConfig):
    global FLAGS
    FLAGS = cfg

    if not os.path.isabs(FLAGS.savedir):
        FLAGS.savedir = os.path.join(hydra.utils.get_original_cwd(), FLAGS.savedir)

    logging.info("flags:\n%s\n", pprint.pformat(dict(FLAGS)))

    if record.symlink_path(FLAGS.savedir, os.path.join(hydra.utils.get_original_cwd(), "latest")):
        logging.info("savedir: %s (symlinked as 'latest')", FLAGS.savedir)
    else:
        logging.info("savedir: %s", FLAGS.savedir)

    train_id = "%s/%s/%s" % (
        FLAGS.entity if FLAGS.entity is not None else getpass.getuser(),
        FLAGS.project,
        FLAGS.group,
    )

    logging.info("train_id: %s", train_id)

    if FLAGS.use_moolib_envpool:
        EnvPoolCls = moolib.EnvPool  # only supports discrete action space curently
    else:
        EnvPoolCls = EnvPool

    create_env_fn = __MakeEnv__.build(FLAGS.make_env_name, FLAGS.env)

    envs = EnvPoolCls(
        create_env_fn,
        num_processes=FLAGS.num_actor_processes,
        batch_size=FLAGS.actor_batch_size,
        num_batches=FLAGS.num_actor_batches,
    )

    logging.info(f"EnvPool started: {envs}")

    dummy_env = create_env_fn()  # TODO: pass an option `dummy_env=True`, so that only the desired attributes can be accessed
    observation_space = dummy_env.observation_space
    action_space = dummy_env.action_space
    info_keys_custom = getattr(dummy_env, "info_keys_custom", None)
    dummy_env.close()
    del dummy_env
    logging.info(f"observation_space: {observation_space}")
    logging.info(f"action_space: {action_space}")

    model = create_model(observation_space, action_space)
    model.to(device=FLAGS.device)
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    learner_state = LearnerState(model, optimizer, scheduler)

    model_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of model parameters: %i", model_numel)
    record.write_metadata(
        FLAGS.localdir,
        hydra.utils.get_original_cwd(),
        flags=omegaconf.OmegaConf.to_container(FLAGS),
        model_numel=model_numel,
    )

    if FLAGS.wandb:
        import wandb

        wandb.init(
            project=str(FLAGS.project),
            config=omegaconf.OmegaConf.to_container(FLAGS),
            group=FLAGS.group,
            entity=FLAGS.entity,
            name=FLAGS.local_name,
            tags=FLAGS.tags,
            notes=FLAGS.notes,
        )

    zero_action = sample_from_space(action_space, batch_size=FLAGS.actor_batch_size, to_torch_tensor=True)
    env_states = [
        EnvBatchState(
            FLAGS,
            model,
            zero_action,
            info_keys_custom=info_keys_custom,
        )
        for _ in range(FLAGS.num_actor_batches)
    ]

    rpc = moolib.Rpc()
    rpc.set_name(FLAGS.local_name)
    rpc.connect(FLAGS.connect)

    rpc_group = moolib.Group(rpc, name=train_id)

    accumulator = moolib.Accumulator(
        group=rpc_group,
        name="model",
        parameters=model.parameters(),
        buffers=model.buffers(),
    )
    accumulator.set_virtual_batch_size(FLAGS.virtual_batch_size)

    learn_batcher = moolib.Batcher(FLAGS.batch_size, FLAGS.device, dim=1)

    stats = {
        "SPS": common.StatMean(),
        "env_act_steps": common.StatSum(),
        "env_train_steps": common.StatSum(),
        "optimizer_steps": common.StatSum(),
        "steps_done": common.StatSum(),
        "episodes_done": common.StatSum(),
        #
        "mean_episode_return": common.StatMean(),
        "mean_episode_step": common.StatMean(),
        "running_reward": common.StatMean(),
        "running_step": common.StatMean(),
        "end_episode_success": common.StatMean(),
        "end_episode_progress": common.StatMean(),
        #
        "unclipped_grad_norm": common.StatMean(),
        "model_version": common.StatSum(),
        "virtual_batch_size": common.StatMean(),
        "num_gradients": common.StatMean(),
        #
        "entropy_loss": common.StatMean(),
        "pg_loss": common.StatMean(),
        "baseline_loss": common.StatMean(),
        "kl_loss": common.StatMean(),
        "total_loss": common.StatMean(),
    }
    if info_keys_custom is not None:
        for k in info_keys_custom:
            stats[f"end_{k}"] = common.StatMean()
    learner_state.global_stats = copy.deepcopy(stats)

    checkpoint_path = os.path.join(FLAGS.savedir, "checkpoint.tar")

    if os.path.exists(checkpoint_path):
        logging.info("Loading checkpoint: %s" % checkpoint_path)
        load_checkpoint(checkpoint_path, learner_state)
        accumulator.set_model_version(learner_state.model_version)
        logging.info("loaded stats %s", learner_state.global_stats)

    global_stats_accumulator = common.GlobalStatsAccumulator(rpc_group, learner_state.global_stats)

    terminate = False
    previous_signal_handler = {}

    def signal_handler(signum, frame):
        nonlocal terminate
        logging.info(
            "Got signal %s, quitting!",
            signal.strsignal(signum) if hasattr(signal, "strsignal") else signum,
        )
        terminate = True
        previous_handler = previous_signal_handler[signum]
        if previous_handler is not None:
            previous_signal_handler[signum] = None
            signal.signal(signum, previous_handler)

    previous_signal_handler[signal.SIGTERM] = signal.signal(signal.SIGTERM, signal_handler)
    previous_signal_handler[signal.SIGINT] = signal.signal(signal.SIGINT, signal_handler)

    if torch.backends.cudnn.is_available():
        logging.info("Optimising CuDNN kernels")
        torch.backends.cudnn.benchmark = True

    # Run.
    now = time.time()
    warm_up_time = FLAGS.warmup
    prev_env_train_steps = 0
    prev_global_env_train_steps = 0
    next_env_index = 0
    last_log = now
    last_reduce_stats = now
    is_leader = False
    is_connected = False
    while not terminate:
        prev_now = now
        now = time.time()

        steps = learner_state.global_stats["env_train_steps"].result()
        if steps >= FLAGS.total_steps:
            logging.info("Stopping training after %i steps", steps)
            break

        rpc_group.update()
        accumulator.update()
        if accumulator.wants_state():
            assert accumulator.is_leader()
            accumulator.set_state(learner_state.save())
        if accumulator.has_new_state():
            assert not accumulator.is_leader()
            learner_state.load(accumulator.state())

        was_connected = is_connected
        is_connected = accumulator.connected()
        if not is_connected:
            if was_connected:
                logging.warning("Training interrupted!")
            # If we're not connected, sleep for a bit so we don't busy-wait
            logging.info("Your training will commence shortly.")
            time.sleep(1)
            continue

        was_leader = is_leader
        is_leader = accumulator.is_leader()
        if not was_connected:
            logging.info(
                "Training started. Leader is %s, %d members, model version is %d"
                % (
                    "me!" if is_leader else accumulator.get_leader(),
                    len(rpc_group.members()),
                    learner_state.model_version,
                )
            )
            prev_global_env_train_steps = learner_state.global_stats["env_train_steps"].result()

            if warm_up_time > 0:
                logging.info("Warming up for %g seconds", warm_up_time)

        if warm_up_time > 0:
            warm_up_time -= now - prev_now

        learner_state.train_time += now - prev_now
        if now - last_reduce_stats >= 2:
            last_reduce_stats = now
            # NOTE: If getting "TypeError: unsupported operand type(s) for -: 'float' and 'StatMean'"
            # then probably assigning with `stats["key"] = value`. Use `stats["key"] += value` instead.
            global_stats_accumulator.reduce(stats)
        if now - last_log >= FLAGS.log_interval:
            delta = now - last_log
            last_log = now

            global_stats_accumulator.reduce(stats)
            global_stats_accumulator.reset()

            prev_env_train_steps = calculate_sps(stats, delta, prev_env_train_steps, is_global=False)
            prev_global_env_train_steps = calculate_sps(
                learner_state.global_stats, delta, prev_global_env_train_steps, is_global=True
            )

            steps = learner_state.global_stats["env_train_steps"].result()

            log(stats, step=steps, is_global=False, wandb=FLAGS.wandb)
            log(learner_state.global_stats, step=steps, is_global=True, wandb=FLAGS.wandb)

            if warm_up_time > 0:
                logging.info("Warming up up for an additional %g seconds", round(warm_up_time))

        if is_leader:
            if not was_leader:
                leader_filename = os.path.join(FLAGS.savedir, "leader-%03d" % learner_state.num_previous_leaders)
                record.symlink_path(FLAGS.localdir, leader_filename)
                logging.info("Created symlink %s -> %s", leader_filename, FLAGS.localdir)
                learner_state.num_previous_leaders += 1
            if not was_leader and not os.path.exists(checkpoint_path):
                logging.info("Training a new model from scratch.")
            if learner_state.train_time - learner_state.last_checkpoint >= FLAGS.checkpoint_interval:
                learner_state.last_checkpoint = learner_state.train_time
                save_checkpoint(checkpoint_path, learner_state, FLAGS)
            if learner_state.train_time - learner_state.last_checkpoint_history >= FLAGS.checkpoint_history_interval:
                learner_state.last_checkpoint_history = learner_state.train_time
                save_checkpoint(
                    os.path.join(
                        FLAGS.savedir,
                        "checkpoint_v%d.tar" % learner_state.model_version,
                        FLAGS,
                    ),
                    learner_state,
                )

        if accumulator.has_gradients():
            gradient_stats = accumulator.get_gradient_stats()
            stats["virtual_batch_size"] += gradient_stats["batch_size"]
            stats["num_gradients"] += gradient_stats["num_gradients"]
            step_optimizer(learner_state, stats)
            accumulator.zero_gradients()
        elif not learn_batcher.empty() and accumulator.wants_gradients():
            compute_gradients(FLAGS, learn_batcher.get(), learner_state, stats)
            accumulator.reduce_gradients(FLAGS.batch_size)
        else:
            if accumulator.wants_gradients():
                accumulator.skip_gradients()

            # Generate data.
            cur_index = next_env_index
            next_env_index = (next_env_index + 1) % FLAGS.num_actor_batches

            env_state = env_states[cur_index]
            if env_state.future is None:  # need to initialize
                env_state.future = envs.step(cur_index, env_state.prev_action)
            cpu_env_outputs = env_state.future.result()

            env_outputs = nest.map(lambda t: t.to(FLAGS.device, copy=True), cpu_env_outputs)

            env_outputs["prev_action"] = env_state.prev_action
            prev_core_state = env_state.core_state
            model.eval()
            with torch.no_grad():
                actor_outputs, env_state.core_state = model(
                    nest.map(lambda t: t.unsqueeze(0), env_outputs),
                    env_state.core_state,
                )
            actor_outputs = nest.map(lambda t: t.squeeze(0), actor_outputs)
            action = actor_outputs["action"]
            env_state.update(cpu_env_outputs, action, stats)
            # envs.step invalidates cpu_env_outputs
            del cpu_env_outputs
            env_state.future = envs.step(cur_index, action)

            stats["env_act_steps"] += action.numel()

            last_data = {
                "env_outputs": env_outputs,
                "actor_outputs": actor_outputs,
            }
            if warm_up_time <= 0:
                env_state.time_batcher.stack(last_data)

            if not env_state.time_batcher.empty():
                data = env_state.time_batcher.get()
                data["initial_core_state"] = env_state.initial_core_state
                learn_batcher.cat(data)

                # We need the last entry of the previous time batch
                # to be put into the first entry of this time batch,
                # with the initial_core_state to match
                env_state.initial_core_state = prev_core_state
                env_state.time_batcher.stack(last_data)
    if is_connected and is_leader:
        save_checkpoint(checkpoint_path, learner_state, FLAGS)
    logging.info("Graceful exit. Bye bye!")


# Override config_path via --config_path.
@hydra.main(config_path=".", config_name="config")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    # moolib.set_log_level("debug")
    main()

# see https://github.com/facebookresearch/moolib/tree/main/examples#fully-fledged-vtrace-agent
# for usage
