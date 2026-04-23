#!/usr/bin/env bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd /root/openvla || exit 1

export HF_HOME=/root/autodl-tmp/cache
export PYTHONPATH=/root/openvla/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

source /etc/network_turbo
