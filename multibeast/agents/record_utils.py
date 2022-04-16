import logging
import os
import pprint

import moolib
import omegaconf
import torch
from moolib.examples.common import record


def log(stats, step, is_global=False, wandb=False):
    stats_values = {}
    prefix = "global/" if is_global else "local/"
    for k, v in stats.items():
        stats_values[prefix + k] = v.result()
        v.reset()

    logging.info(f"\n{pprint.pformat(stats_values)}")
    if not is_global:
        record.log_to_file(**stats_values)

    if wandb:
        import wandb

        wandb.log(stats_values, step=step)


def save_checkpoint(checkpoint_path, learner_state, flags):
    tmp_path = "%s.tmp.%s" % (checkpoint_path, moolib.create_uid())

    logging.info("saving global stats %s", learner_state.global_stats)

    checkpoint = {
        "learner_state": learner_state.save(),
        "flags": omegaconf.OmegaConf.to_container(flags),
    }

    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    logging.info("Checkpoint saved to %s", checkpoint_path)


def load_checkpoint(checkpoint_path, learner_state):
    checkpoint = torch.load(checkpoint_path)
    learner_state.load(checkpoint["learner_state"])


def calculate_sps(stats, delta, prev_steps, is_global=False):
    env_train_steps = stats["env_train_steps"].result()
    prefix = "global/" if is_global else "local/"
    logging.info("%s calculate_sps %g steps in %g seconds", prefix, env_train_steps - prev_steps, delta)
    stats["SPS"] += (env_train_steps - prev_steps) / delta
    return env_train_steps
