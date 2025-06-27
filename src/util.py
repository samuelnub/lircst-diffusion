import os
import time

def generate_directory_name(model_name, preexisting: str|None=None):
    timestamp = int(time.time())
    return f"../models/{model_name}/{timestamp if preexisting is None else preexisting}/", timestamp

def get_latest_ckpt(model_name, latest_dir: str|None=None):
    model_dir = f"../models/{model_name}/"
    if not os.path.exists(model_dir):
        return None
    directories = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not directories:
        return None

    latest_dir = latest_dir if latest_dir is not None else sorted(directories)[-1]

    lightning_logs_dir = f'{model_dir}{latest_dir}/lightning_logs/'

    ckpt_dir = f'{model_dir}{latest_dir}/lightning_logs/{sorted(os.listdir(lightning_logs_dir))[-1]}/checkpoints/'

    ckpt_filename = os.listdir(ckpt_dir)[0] if os.path.exists(ckpt_dir) else None
    if ckpt_filename is None:
        print(f"No checkpoint found for {model_name} in {ckpt_dir}")
        return None
    return f'{ckpt_dir}{ckpt_filename}', latest_dir