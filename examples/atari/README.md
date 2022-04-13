# Atari

```bash

pip install gym[atari, accept-rom-license]

python -m examples.atari.impala connect=$BROKER_IP:$BROKER_PORT \
    savedir=${PWD}/outputs/multibeast-atari/savedir \
    project=multibeast-atari \
    group=Zaxxon-Breakout \
    env.name=ALE/Breakout-v5 \
    wandb=true \
    num_actor_cpus=64 \
    --config-path=${PWD}/third_party/moolib/examples/vtrace/ \
    --config-name=config
```
