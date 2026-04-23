source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/openvla/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

source /etc/network_turbo
cd /root/openvla
