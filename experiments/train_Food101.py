import os
import datetime
import argparse
import sys
import logging
from abc import ABC, abstractmethod

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import v2

import lightning as L
import torchmetrics
from lightning.pytorch import seed_everything, loggers

from vit_pytorch import ViT
from SANIA import SANIA_AdamSQR, SANIA_AdagradSQR, KATE

try:
    from lightning.pytorch.accelerators import find_usable_cuda_devices
except ImportError:
    from lightning_fabric.accelerators import find_usable_cuda_devices

import wandb

from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class Food101DataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = os.getenv("TORCHVISION_DATASETS_DIR"), batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 2
        self.num_labels = 101
        self.class_names = None

        self.transform_train = v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.transform_val = v2.Compose([
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def prepare_data(self):
        torchvision.datasets.Food101(self.data_dir, split="train", download=True)
        torchvision.datasets.Food101(self.data_dir, split="test", download=True)

    def setup(self, stage: str):

        if stage in ('fit', None):
            self.train_dataset = torchvision.datasets.Food101(self.data_dir, split="train", download=False, transform=self.transform_train)
            self.val_dataset = torchvision.datasets.Food101(self.data_dir, split="test", download=False, transform=self.transform_val)
            self.class_names = self.train_dataset.classes
        if stage in ('test', 'predict'):
            self.val_dataset = torchvision.datasets.Food101(self.data_dir, split="test", download=False, transform=self.transform_val)
            if self.class_names is None:
                self.class_names = self.val_dataset.classes

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


optimizers_dict = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SANIA_AdamSQR': SANIA_AdamSQR,
    'SANIA_AdagradSQR': SANIA_AdagradSQR,
    'KATE': KATE,
    'Adagrad': torch.optim.Adagrad,
}


class Food101Classifier(L.LightningModule):

    def __init__(self, num_labels: int, config: dict):
        super().__init__()

        self.num_labels = num_labels

        # record hyperparams for loggers
        self.save_hyperparameters({
            'dataset': 'Food101',
            'task': 'multi-class-classification',
            'model': 'ViT',
            'config': config,
        })

        self.optimizer_name = config['optimizer']
        self.optimizer_hparams = config['optimizer_hparams']

        self.SANIA = False
        if self.optimizer_name in ("SANIA_AdamSQR", "SANIA_AdagradSQR"):
            self.SANIA = True
            self.automatic_optimization = False

        self.model = self.build_model()

        self.loss_fn = self.define_loss_fn()
        self.val_acc = self.define_val_acc_metric()

    def define_loss_fn(self, *args, **kwargs):
        return nn.CrossEntropyLoss()

    def define_val_acc_metric(self, *args, **kwargs):
        return torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_labels, top_k=1)

    def build_model(self, *args, **kwargs):
        # Pull optional ViT hparams from config, else use sensible defaults
        vit_cfg = self.hparams['config'].get('vit', {})
        image_size = vit_cfg.get('image_size', 224)
        patch_size = vit_cfg.get('patch_size', 16)
        dim = vit_cfg.get('dim', 768)         # base width
        depth = vit_cfg.get('depth', 12)      # base depth
        heads = vit_cfg.get('heads', 12)
        mlp_dim = vit_cfg.get('mlp_dim', 3072)
        dropout = vit_cfg.get('dropout', 0.1)
        emb_dropout = vit_cfg.get('emb_dropout', 0.1)

        model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=self.num_labels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        return model.to(DEVICE)

    def unpack_batch(self, batch):
        x, y = batch
        return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

    def training_step(self, batch):
        x, y = self.unpack_batch(batch)
        logits = self.model(x)

        if self.SANIA:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            closure = lambda: self.loss_fn(logits, y)
            loss = closure()
            loss.backward()
            optimizer.step(closure=closure)
        else:
            loss = self.loss_fn(logits, y)
            
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.log('lr', self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)

    def validation_step(self, batch):
        x, y = self.unpack_batch(batch)

        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.val_acc is not None:
            self.val_acc(logits, y)
            self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = self.hparams['config']['optimizer']
        optimizer_hparams = self.hparams['config']['optimizer_hparams']
        optimizer = optimizers_dict[optimizer](self.model.parameters(), **optimizer_hparams)
        
        return [optimizer]


def run_experiment(config: dict) -> None:

    data_module = Food101DataModule(batch_size=config['batch_size'])
    data_module.prepare_data()
    data_module.setup('fit')

    seed_everything(config['seed'], workers=True)

    model = Food101Classifier(num_labels=data_module.num_labels, config=config)

    csv_logger = loggers.CSVLogger(
        save_dir=f"logs/{model.hparams['dataset']}",
        version=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    
    import wandb
    wandb.login()  # will open a prompt the first time
    RUN_NAME    = f"vit-food101-{config['optimizer']}-{config['optimizer_hparams']}"
    PROJECT     = "SANIA-Food101"
    SAVE_DIR    = "wandb/food101-vit"
    CKPT_DIR    = "ckpts/food101-vit"

    wandb_logger = loggers.WandbLogger(
        project=PROJECT,
        name=RUN_NAME,
        save_dir=SAVE_DIR,
    )

    wandb_logger.experiment.config.update(
        {
            'dataset/name': model.hparams['dataset'],
            'model': model.hparams['model'],
            'config': config,
            'task': model.hparams['task'],
        }
    )
    print(config)

    gpu = find_usable_cuda_devices(1)
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        logger=[csv_logger, wandb_logger],
        accelerator='gpu',
        devices=[5,],
        log_every_n_steps=min(len(data_module.train_dataloader()), 50)
        )

    trainer.fit(model=model, datamodule=data_module)

    wandb.finish()


def main(args):

    if args.seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [args.seed]

    for seed in seeds:

        config = {
            'seed': seed,
            'max_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'optimizer': args.optimizer,
            'optimizer_hparams': args.optimizer_hparams,
        }

        print(args)

        if 'lr' in args.optimizer_hparams and args.optimizer_hparams.get('lr') == -1:
            print("[INFO]: Learning rate sweep is enabled.")
            # lrs = [10**x for x in range(-10, 6)]
            lrs = [10**x for x in range(-5, 3)]

            for lr in lrs:
                config['optimizer_hparams']['lr'] = float(lr)
                run_experiment(config=config)
        elif 'eta_max' in args.optimizer_hparams and args.optimizer_hparams.get('eta_max') == -1:
            print("[INFO]: Learning rate sweep is enabled.")
            # lrs = [10**x for x in range(-10, 6)]
            eta_range = [10**x for x in range(-5, 3)]

            for eta_max in eta_range:
                config['optimizer_hparams']['eta_max'] = float(eta_max)
                run_experiment(config=config)
        elif 'max_lr' in args.optimizer_hparams and args.optimizer_hparams.get('max_lr') == -1:
            print("[INFO]: Learning rate sweep is enabled.")
            lr_range = [10**x for x in range(-5, 3)]

            for max_lr in lr_range:
                config['optimizer_hparams']['max_lr'] = float(max_lr)
                run_experiment(config=config)
        else:
            run_experiment(config=config)


def parse_optimizer_hparams(unknown):
        opt_hparams = {}
        for arg in unknown:
            hparam = arg.replace(' ', '=')
            key, value = hparam.split("=")
            key = key.lstrip('-')
            try:
                if '.' in value or 'e' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            opt_hparams[key] = value
        return opt_hparams

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Provide additional arguments to parser, such as optimizer hyperparameters, after required arguments.")
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")

    args, unknown = parser.parse_known_args()

    args.optimizer_hparams = parse_optimizer_hparams(unknown)

    main(args)