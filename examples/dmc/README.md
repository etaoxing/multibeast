# dm_control

```bash
# use dmc2gym since they already convert from dm_control format to gym
pip install git+https://github.com/denisyarats/dmc2gym.git

export MUJOCO_GL="egl"
export EGL_DEVICE_ID=0

python -m examples.dmc.impala connect=$BROKER_IP:$BROKER_PORT \
    savedir=${PWD}/outputs/multibeast-dmc/savedir \
    project=multibeast-dmc \
    group=G0-cheetah-run \
    env.domain_name=cheetah \
    env.task_name=run \
    env.from_pixels=false \
    optimizer.learning_rate=3e-4 \
    wandb=0
```
