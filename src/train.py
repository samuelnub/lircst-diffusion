# Full pipeline
from encoded_conditional_diffusion import ECDiffusion
from util import *

# Setup Diffusion modules
import pytorch_lightning as pl
from Diffusion.EMA import EMA
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

pre_load: bool = True # Load the latest checkpoint if available
train_mode: bool = True
test_afterward: bool = False

def train():
    for name, model_arg in model_args.items():
        print(f"Training {name}...")

        train_dataset, valid_dataset, test_dataset = get_dataset()

        model = ECDiffusion(**model_arg, 
                            train_dataset=train_dataset, 
                            valid_dataset=valid_dataset, 
                            test_dataset=test_dataset)

        # TODO: Bugged if pre_load is True
        default_root_dir, timestamp = generate_directory_name(name, get_latest_ckpt(name)[1] if pre_load else None)
        
        wandb_config = {
            "name": name,
            "physics": model_arg["physics"],
            "latent": model_arg["latent"],
            "timestamp": timestamp,
        }
        wandb_project = "lircst-diffusion"
        with wandb.init(project=wandb_project, config=wandb_config):
            print(f"Initialized wandb for {name} with config: {wandb_config}")

            trainer = pl.Trainer(
                max_epochs=101,
                max_steps=2e5,
                callbacks=[EMA(0.9999)],
                accelerator='gpu',
                devices=[0],
                num_sanity_val_steps=0,  # Disable sanity check on dataloader
                limit_val_batches=4,
                default_root_dir=default_root_dir,
            )
            
            trainer.fit(model, ckpt_path=get_latest_ckpt(name)[0] if pre_load else None)
            
            if test_afterward:
                trainer.test(model, ckpt_path=get_latest_ckpt(name)[0] if pre_load else None)

if train_mode:
    train()
