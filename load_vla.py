import os
from pathlib import Path
import yaml
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_env(cfg):
    os.environ.setdefault("HF_HOME", cfg["runtime"]["hf_home"])
    os.environ.setdefault("HF_ENDPOINT", cfg["runtime"]["hf_endpoint"])


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device):
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_openvla(cfg, device):
    model_name = cfg["model"]["name"]
    dtype = get_dtype(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        torch_dtype=dtype,
        low_cpu_mem_usage=cfg["model"]["low_cpu_mem_usage"],
        attn_implementation=cfg["model"]["attn_implementation"],
    )

    model = model.to(device)
    model.eval()
    return processor, model


def run_inference(cfg, processor, model, device):
    image_path = cfg["inference"]["image_path"]
    instruction = cfg["inference"]["instruction"]
    unnorm_key = cfg["inference"]["unnorm_key"]

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    inputs = processor(prompt, image).to(
        device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
        )

    return action


def main():
    cfg = load_config("config.yaml")
    setup_env(cfg)

    device = get_device()
    processor, model = load_openvla(cfg, device)
    print("model loaded successfully")

    action = run_inference(cfg, processor, model, device)
    print("predicted action:", action)


if __name__ == "__main__":
    main()