<!-- start about -->

[pypi-url]: https://pypi.python.org/pypi/multibeast
[license-badge]: https://img.shields.io/pypi/l/multibeast.svg
[version-badge]: https://img.shields.io/pypi/v/multibeast.svg
[pyversion-badge]: https://img.shields.io/pypi/pyversions/multibeast.svg

[tests-badge]: https://github.com/etaoxing/multibeast/actions/workflows/tests.yml/badge.svg
[tests-url]: https://github.com/etaoxing/multibeast/actions/workflows/tests.yml

[docs-badge]: https://img.shields.io/readthedocs/multibeast.svg
[docs-url]: https://multibeast.readthedocs.io/

# 🦬 MultiBeast

[![license][license-badge]][pypi-url]
[![version][version-badge]][pypi-url]
[![pyversion][pyversion-badge]][pypi-url]
[![tests][tests-badge]][tests-url]
[![docs][docs-badge]][docs-url]

A research codebase for asynchronous RL.

<!-- end about -->

---

<!-- start quickstart -->

# Quickstart

`pip install multibeast` provides:

🐄 Asynchronous, distributed RL algorithms based on the IMPALA implementation in [moolib](https://github.com/facebookresearch/moolib), the successor to [`torchbeast`](https://github.com/facebookresearch/torchbeast).

🐂 Training results tracked and available on wandb, with hydra config files, for reproducibility.

🐃 An avoidance of inheritance & complexity, to facilitate fast research iteration. Core implementation details are kept as simple and self-contained as possible, in one folder.

<!-- end quickstart -->

<!-- start example -->

See [`examples/`](examples/) on how to use `multibeast` in your code.

<!-- end example -->

# Setup from source

## Install moolib

```bash
# Setup conda env
conda create --name mb python=3.7
conda activate mb
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install moolib
conda install -c conda-forge cmake==3.14.5  # need cmake>=3.14 for moolib
cd third_party/moolib/
USE_CUDA=1 python setup.py build --debug install  # or pip install -e .
```

### Compiling in Docker


If you are having issues building moolib, then try compiling inside a Docker container.
```bash
export CONDA_PATH=~/miniconda3
export REPO_PATH=$PWD

docker pull nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
docker run -it \
  -v $REPO_PATH:$REPO_PATH \
  -v $CONDA_PATH:$CONDA_PATH \
  -e REPO_PATH=${REPO_PATH} \
  -e CONDA_PATH=${CONDA_PATH} \
  --gpus all \
  --name mb \
  nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04 bash

# inside container
export PATH=${CONDA_PATH}/bin:$PATH
cd $REPO_PATH

# then go back and install moolib
```

If you run into the following error, then try cloning `moolib/third_party/` submodule repos first:
```bash
CMake Error at CMakeLists.txt:51 (pybind11_add_module):
  Unknown CMake command "pybind11_add_module".
```

If cudnn is missing then just try reinstalling cuda:
```bash
apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    git \
    vim \
    wget \
    pkg-config \
    software-properties-common \
  && rm -rf /var/lib/apt/lists/*

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin \
  && mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
  && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
  && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
apt-get update \
  && apt-get -y install cuda-toolkit-11-4 \
  && apt-get clean && rm -rf /var/lib/apt/lists/* 
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Verify moolib:

```bash
pip install autorom[accept-rom-license]
pip install gym[atari]

python -m moolib.broker  # Note that a **single broker is enough** for all your experiments.
export BROKER_PORT=4431  # Should match `Broker listening at 0.0.0.0:4431`
export BROKER_IP=$(echo $SSH_CONNECTION | cut -d' ' -f3)  # Should give your machine's IP.
export BROKER_IP=0.0.0.0  # can also use local IP address if broker if on same machine

python -m examples.vtrace.experiment connect=$BROKER_IP:$BROKER_PORT \
    wandb=0 \
    savedir=outputs/moolib-atari/savedir \
    project=moolib-atari \
    group=Zaxxon-Breakout \
    env.name=ALE/Breakout-v5

# To add more peers to this experiment, start more processes with the
# same `project` and `group` settings, using a different setting for
# `device` (default: `'cuda:0'`).
```

Try `pip uninstall gym atari-py ale-py && pip install gym[atari]` if you run into the following error:
```bash
Error in env: ModuleNotFoundError: No module named 'gym.envs.atari'
```
[[link]](https://github.com/openai/gym/issues/2498#issuecomment-984996272)

If you run into the following error, then try using the local IP address (0.0.0.0 or 127.0.0.1):
```python
terminate called after throwing an instance of 'std::runtime_error'
  what():  In connectFromLoop at .../moolib/src/tensorpipe/tensorpipe/transport/uv/uv.h:313 "rv < 0: network is unreachable"
Aborted (core dumped)
```
[[link]](https://github.com/facebookresearch/moolib/issues/34)

## Install multibeast

```bash
pip install -e .
```

