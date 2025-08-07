import keys
import wandb
wandb.login(key=keys.wandb)

# Full pipeline
from encoded_conditional_diffusion import ECDiffusion
from util import generate_directory_name, get_latest_ckpt, model_args, get_dataset

# Setup Diffusion modules
import gc
import torch
import pytorch_lightning as pl
from Diffusion.EMA import EMA
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

pre_load: bool = True # Load the latest checkpoint if available
train_mode: bool = False
test_afterward: bool = True

dataset_train, dataset_val, dataset_test = get_dataset()


def train():
    for name, model_arg in model_args.items():
        print(f"Training {name}...")

        model = ECDiffusion(
            train_dataset=dataset_train,
            valid_dataset=dataset_val,
            test_dataset=dataset_test,
            **model_arg
        )

        default_root_dir, timestamp = generate_directory_name(name, get_latest_ckpt(name)[1] if pre_load else None)
        
        wandb_config = {
            "name": name,
            "physics": model_arg["physics"],
            "latent": model_arg["latent"],
            "predict_mode": model_arg["predict_mode"],
            "condition_A_T": model_arg["condition_A_T"],
            "degradation": model_arg["degradation"],
            "timestamp": timestamp,
        }
        wandb_project = "lircst-diffusion"
        with wandb.init(project=wandb_project, config=wandb_config):
            print(f"Initialized wandb for {name} with config: {wandb_config}")

            wandb.define_metric("test/psnr_scat", summary="mean")
            wandb.define_metric("test/ssim_scat", summary="mean")
            wandb.define_metric("test/rmse_scat", summary="mean")
            wandb.define_metric("test/psnr_atten", summary="mean")
            wandb.define_metric("test/ssim_atten", summary="mean")
            wandb.define_metric("test/rmse_atten", summary="mean")

            wandb.define_metric("data/degradation_snr", summary="mean")
            wandb.define_metric("data/degradation_psnr", summary="mean")

            trainer = pl.Trainer(
                #detect_anomaly=True, # Enable anomaly detection for debugging
                max_epochs=1, # Experiments seem to show that it converges around 100-120 epochs
                limit_train_batches=10, # TODO, as well as epoch max
                max_steps=2e5,
                callbacks=[EMA(0.9999)],
                accelerator='gpu',
                devices=[0],
                num_sanity_val_steps=0,  # Disable sanity check on dataloader
                limit_val_batches=4, # Limit validation batches for faster training
                default_root_dir=default_root_dir,
            )
            
            if train_mode:
                trainer.fit(model, ckpt_path=get_latest_ckpt(name)[0] if pre_load else None)
            
            if test_afterward:
                trainer.test(model, ckpt_path=get_latest_ckpt(name)[0] if pre_load else None)

        # Free up memory
        print(f"Finished training {name}. Cleaning up...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

train()
