import json
import wandb
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import HandwritingDataset
from handwriting_model import HandwritingAuthorAdaptation

wandb.login()

class HandwritingTrainer():
    def __init__(self, config):
        self.config = config

        self.logger = WandbLogger(
            project=self.config.project_name,
            log_model='all'
        )

        self.model = HandwritingAuthorAdaptation(lr=self.config.lr, optim=self.config.optim, dictionary_size=self.config.dictionary_size)
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            monitor="val_loss",
            mode="min"
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        seed_everything(5340, workers=True)
    
    def train(self):
        train_dataset = HandwritingDataset(split='train')
        val_dataset = HandwritingDataset(split='val')

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        trainer = Trainer(
            logger=self.logger,
            callbacks=[self.checkpoint_callback],
            max_epochs=self.config.epochs,
            log_every_n_steps=10,
            accelerator='gpu',
            devices=[0],
            deterministic=True
        )

        trainer.fit(self.model, train_loader, val_loader)
        self.checkpoint_callback.best_model_path

def main():
    """
    Main function to run when the script is initialised
    """
    with open("train_config.json") as f:
        run_config = json.load(f)

    wandb.init(
        project=run_config["project_name"],
        group=run_config["group_name"],
        config=run_config,
        save_code=True)

    config = wandb.config

    wandb.run.name = run_config["run_name"]
    wandb.run.save()

    trainer = HandwritingTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()