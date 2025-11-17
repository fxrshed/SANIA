import os
import sys
import datetime
import logging
import argparse
import math
from dataclasses import dataclass

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import v2

import lightning as L
import torchmetrics
from lightning.pytorch import seed_everything, loggers, callbacks

from datasets import load_dataset, DatasetDict
import tiktoken
import wandb

from SANIA import SANIA_AdamSQR, SANIA_AdagradSQR, KATE

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class SampleGenerationCallback(callbacks.Callback):
    def __init__(
        self,
        prompts=("The meaning of life", "Once upon a time"),
        max_new_tokens=60,
        temperature=0.9,
        top_k=50,
        every_n_steps=None,
        table_key="gen/samples",
    ):
        super().__init__()
        self.prompts = list(prompts)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.every_n_steps = every_n_steps
        self.table_key = table_key
        self.tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def _generate_and_log(self, trainer: L.Trainer, pl_module: L.LightningModule, tag: str):
        model = pl_module.model
        model.eval()

        rows = []
        for p in self.prompts:
            inp = self.tok(p, return_tensors="pt").to(pl_module.device)
            out_ids = model.generate(
                **inp,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
            )
            txt = self.tok.decode(out_ids[0], skip_special_tokens=True)
            rows.append([p, txt])

        table = wandb.Table(columns=["prompt", "sample"], data=rows)
        if isinstance(trainer.logger, loggers.WandbLogger):
            trainer.logger.experiment.log({f"{self.table_key}/{tag}": table, "global_step": trainer.global_step})

    def on_validation_epoch_end(self, trainer, pl_module):
        self._generate_and_log(trainer, pl_module, tag=f"epoch{trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps and trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            self._generate_and_log(trainer, pl_module, tag=f"step{trainer.global_step}")

IGNORE_INDEX = -100

class WikiTextDataModule(L.LightningDataModule):
    def __init__(
        self,
        hf_name: str = "Salesforce/wikitext",
        hf_config: str = "wikitext-103-raw-v1",
        text_col: str = "text",
        block_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 2,
        train_split: str = "train",
        val_split: str = "validation",
        labels_style: str = "hf",  # keep "hf" for HF models
    ):
        super().__init__()
        self.hf_name = hf_name
        self.hf_config = hf_config
        self.text_col = text_col
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.labels_style = labels_style

        self.tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.tok.pad_token = self.tok.eos_token
        self.vocab_size = self.tok.vocab_size

    def prepare_data(self):
        load_dataset(self.hf_name, self.hf_config)

    def setup(self, stage=None):
        raw = load_dataset(self.hf_name, self.hf_config)

        if self.val_split not in raw:
            if "validation" in raw:
                self.val_split = "validation"
            elif "val" in raw:
                raw = DatasetDict(train=raw["train"], validation=raw["val"])
                self.val_split = "validation"
            else:
                split = raw["train"].train_test_split(test_size=0.01, seed=0)
                raw = DatasetDict(train=split["train"], validation=split["test"])
                self.val_split = "validation"

        def clean_fn(batch):
            texts = [t for t in batch[self.text_col] if isinstance(t, str) and t.strip()]
            return {self.text_col: texts}

        train_clean = raw[self.train_split].map(clean_fn, batched=True, remove_columns=raw[self.train_split].column_names)
        val_clean   = raw[self.val_split].map(clean_fn,   batched=True, remove_columns=raw[self.val_split].column_names)

        # tokenize each row (no truncation)
        def tok_fn(batch):
            out = self.tok(batch[self.text_col], add_special_tokens=False)
            return {"input_ids": out["input_ids"]}

        tok_train = train_clean.map(tok_fn, batched=True, remove_columns=train_clean.column_names)
        tok_val   = val_clean.map(tok_fn,   batched=True, remove_columns=val_clean.column_names)

        # pack into fixed blocks
        def pack_and_chunk(examples):
            flat = np.fromiter((t for seq in examples["input_ids"] for t in seq), dtype=np.int64)
            L = (len(flat) // self.block_size) * self.block_size
            if L == 0:
                return {"input_ids": []}
            flat = flat[:L]
            input_ids = flat.reshape(-1, self.block_size)
            return {"input_ids": input_ids}

        lm_train = tok_train.map(pack_and_chunk, batched=True, batch_size=1000)
        lm_val   = tok_val.map(pack_and_chunk,   batched=True, batch_size=1000)

        lm_train.set_format(type="torch", columns=["input_ids"])
        lm_val.set_format(type="torch", columns=["input_ids"])
        self.train_ds, self.val_ds = lm_train, lm_val

    def collate(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.ones_like(input_ids)
        # HF causal LM computes shift internally when labels are provided
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=self.collate)

optimizers_dict = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SANIA_AdamSQR': SANIA_AdamSQR,
    'SANIA_AdagradSQR': SANIA_AdagradSQR,
    'KATE': KATE,
    'Adagrad': torch.optim.Adagrad,
}

class GPT2Mini_Wikitext_Finetune(L.LightningModule):
    def __init__(self, config: dict, 
                 lr: float = 1e-4, 
                 weight_decay: float = 0.01, 
                 warmup_steps: int = 1000, 
                 max_steps: int = 50_000):
        
        super().__init__()
        self.save_hyperparameters({
            'dataset': 'wikitext-103-raw-v1',
            'task': 'text-generation',
            'model': 'gpt2mini',
            'config': config,
        })

        self.model = AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini")
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.gradient_checkpointing_enable()

        self.optimizer_name = config['optimizer']
        self.optimizer_hparams = config['optimizer_hparams']
        self.SANIA = False
        if self.optimizer_name in ("SANIA_AdamSQR", "SANIA_AdagradSQR"):
            self.SANIA = True
            self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)     # <-- call the HF model directly
        loss = out.loss
        if self.SANIA:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            def closure():
                return loss
            loss = closure()
            loss.backward()
            optimizer.step(closure=closure)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("val/loss", out.loss, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):

        optimizer = self.hparams['config']['optimizer']
        optimizer_hparams = self.hparams['config']['optimizer_hparams']
        
        sched_type = optimizer_hparams['sched_type']
        
        if self.SANIA:
            opt = optimizers_dict[optimizer](self.model.parameters(), **optimizer_hparams)
            return [opt]
        else:
            if optimizer == 'AdamW':
                no_decay = ["bias", "LayerNorm.weight"]
                decay, nodecay = [], []
                for n, p in self.model.named_parameters():
                    if not p.requires_grad: 
                        continue
                    (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
                opt = torch.optim.AdamW(
                    [{"params": decay,   "weight_decay": 0.01},
                    {"params": nodecay, "weight_decay": 0.0}],
                    lr=optimizer_hparams['lr'], betas=(0.9, 0.95),
                )
            else:
                opt = optimizers_dict[optimizer](self.model.parameters(), **optimizer_hparams)

            warmup_steps = optimizer_hparams['warmup_steps']
            max_steps = self.hparams['config']['max_steps']

            if sched_type == 'linear':
                sch = get_linear_schedule_with_warmup(opt, warmup_steps, max_steps)
            elif sched_type == 'cosine':
                sch = get_cosine_schedule_with_warmup(opt, warmup_steps, max_steps)                
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

def main(args):


    wandb.login()

    # --- Run config ---
    PROJECT     = "sania-gpt2mini-finetune"
    RUN_NAME    = f"wikitext-gpt2mini-block256-longrun-{args.}"
    SAVE_DIR    = "wandb/wikitext-gpt2mini"
    CKPT_DIR    = "ckpts/wikitext-gpt2mini"

    BLOCK_SIZE  = 256
    PER_GPU_BS  = 8                     # raise/lower to fit memory
    ACCUM       = 2                     # effective global batch = PER_GPU_BS * num_gpus * ACCUM
    MAX_STEPS   = args.n_steps
    PRECISION   = "bf16-mixed" if torch.cuda.is_available() else "32-true"

    seed_everything(args.seed, workers=True)

    dm = WikiTextDataModule(
        hf_name="Salesforce/wikitext",
        hf_config="wikitext-103-raw-v1",
        block_size=BLOCK_SIZE,
        batch_size=PER_GPU_BS,
        labels_style="hf",            # <- HF model expects labels=input_ids
    )
    dm.prepare_data()

    args.optimizer_hparams['warmpup_steps'] = int(0.02 * MAX_STEPS)

    config = {
        'seed': args.seed,
        'max_steps': args.MAX_STEPS,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'optimizer_hparams': args.optimizer_hparams,
    }

    model = GPT2Mini_Wikitext_Finetune(config=config)

    # --- Logging ---
    wandb_logger = loggers.WandbLogger(
        project=PROJECT,
        name=RUN_NAME,
        save_dir=SAVE_DIR,          # local log folder for this project
    )

    # (Optional) reduce histogram spam on long runs:
    wandb_logger.watch(model, log="gradients", log_freq=1000)

    # Sample generation callback (log occasionally to avoid overhead)
    gen_cb  = SampleGenerationCallback(every_n_steps=4000, max_new_tokens=80)
    ckpt_cb = callbacks.ModelCheckpoint(
        dirpath=CKPT_DIR, filename="{epoch}-{step}-{val_loss:.3f}",
        save_last=True, save_top_k=2, monitor="val/loss", mode="min",
        every_n_train_steps=2000,
    )
    lrmon_cb = callbacks.LearningRateMonitor(logging_interval="step")

    # --- Trainer ---
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        log_every_n_steps=50,
        val_check_interval=2000,
        gradient_clip_val=1.0,
        accumulate_grad_batches=ACCUM,
        precision=PRECISION,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[7,],             # or devices=torch.cuda.device_count()
        strategy="ddp",
        logger=wandb_logger,
        callbacks=[lrmon_cb, ckpt_cb, gen_cb],
        enable_checkpointing=True,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Provide additional arguments to parser, such as optimizer hyperparameters, after required arguments.")
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0, help="Random seed, i.e. --seed=3 will run with seed(3). Enter -1 to train on 5 seeds.")
    
    args, unknown = parser.parse_known_args()
    
    from train_Food101 import parse_optimizer_hparams
    args.optimizer_hparams = parse_optimizer_hparams(unknown)

    main(args)