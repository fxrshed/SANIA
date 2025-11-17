import os
import datetime
import logging
import argparse
import math
from dataclasses import dataclass

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

from llm_baselines.src.models.base import GPTBase

class SampleGenerationCallback(callbacks.Callback):
    """
    Generates text from a few prompts and logs to W&B.
    By default runs at the end of each validation epoch.
    You can also set every_n_steps to log mid-epoch.
    """
    def __init__(
        self,
        prompts=("The meaning of life", "Once upon a time"),
        max_new_tokens=60,
        temperature=0.9,
        top_k=50,
        every_n_steps=None,      # e.g., 500 to log during training
        column_name="samples",
        table_key="gen/samples"
    ):
        super().__init__()
        self.prompts = list(prompts)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.every_n_steps = every_n_steps
        self.column_name = column_name
        self.table_key = table_key

    @torch.no_grad()
    def _generate_and_log(self, trainer: L.Trainer, pl_module: L.LightningModule, tag: str):
        # pl_module.model is your GPTBase instance
        model = pl_module.model
        model.eval()

        rows = []
        for p in self.prompts:
            out = model.generate_from_string(
                p,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )
            rows.append([p, out])

        table = wandb.Table(columns=["prompt", self.column_name], data=rows)
        trainer.logger.experiment.log(
            {f"{self.table_key}/{tag}": table, "global_step": trainer.global_step}
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log once per validation epoch (nice, regular snapshots)
        if isinstance(trainer.logger, loggers.WandbLogger):
            self._generate_and_log(trainer, pl_module, tag=f"epoch{trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Optional mid-epoch logging, if requested
        if self.every_n_steps and trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            if isinstance(trainer.logger, loggers.WandbLogger):
                self._generate_and_log(trainer, pl_module, tag=f"step{trainer.global_step}")


IGNORE_INDEX = -1

class OWTTextDataModule(L.LightningDataModule):
    def __init__(
        self,
        hf_name: str = "Ankursingh/openwebtext_10K",
        text_col: str = "text",
        block_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 2,
        train_split: str = "train",
        val_split: str = "validation",
    ):
        super().__init__()
        self.hf_name = hf_name
        self.text_col = text_col
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

        self.train_split = train_split
        self.val_split = val_split

    def prepare_data(self):
        # triggers download if needed
        load_dataset(self.hf_name)

    def setup(self, stage=None):
        raw = load_dataset(self.hf_name)
        # If dataset has 'val' instead of 'validation', map it
        if self.val_split not in raw:
            # try common alternative names
            alt = "val" if "val" in raw else None
            if alt:
                raw = DatasetDict(train=raw["train"], validation=raw[alt])

        def tok_fn(batch):
            return {"input_ids": [self.enc.encode(t) for t in batch[self.text_col]]}

        tokenized_train = raw[self.train_split].map(tok_fn, batched=True, remove_columns=raw[self.train_split].column_names)
        tokenized_val   = raw["validation"].map(tok_fn,   batched=True, remove_columns=raw["validation"].column_names)

        def pack_and_chunk(examples):
            flat = np.fromiter((tok for seq in examples["input_ids"] for tok in seq), dtype=np.int64)
            L = (len(flat) // self.block_size) * self.block_size
            if L == 0:
                return {"input_ids": [], "labels": []}
            flat = flat[:L]
            input_ids = flat.reshape(-1, self.block_size)
            labels = input_ids.copy()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1]  = IGNORE_INDEX
            return {"input_ids": input_ids, "labels": labels}

        lm_train = tokenized_train.map(pack_and_chunk, batched=True, batch_size=1000)
        lm_val   = tokenized_val.map(pack_and_chunk,   batched=True, batch_size=1000)

        cols = ["input_ids", "labels"]
        self.train_ds = lm_train.with_format("torch", columns=cols)
        self.val_ds   = lm_val.with_format("torch", columns=cols)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False)


class LitGPT(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 2000,
        use_myopt: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build config object for GPTBase
        Cfg = type("Cfg", (), {})  # simple holder
        cfg = Cfg()
        cfg.vocab_size      = vocab_size
        cfg.sequence_length = sequence_length
        cfg.n_layer         = n_layer
        cfg.n_head          = n_head
        cfg.n_embd          = n_embd
        cfg.dropout         = dropout
        cfg.bias            = bias

        self.model = GPTBase(cfg)
        self.example_input_array = torch.zeros(1, sequence_length, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for inference / summaries."""
        out = self.model(x, targets=None, get_logits=True)
        return out["logits"]

    def training_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        logits_loss = self.model(x, targets=y, get_logits=False)
        loss = logits_loss["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        out = self.model(x, targets=y, get_logits=False)
        loss = out["loss"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_after_backward(self):
        # grad-norm diagnostic
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None])
        )
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    def configure_optimizers(self):
        # param groups from your model (decay vs no-decay)
        name_groups = self.model.get_parameter_group_specs()
        param_dict = {n: p for n, p in self.model.named_parameters()}
        optim_groups = []
        for g in name_groups:
            optim_groups.append({
                "params": [param_dict[n] for n in g["params"]],
                **{k: v for k, v in g.items() if k != "params"}
            })

        opt = torch.optim.AdamW(
            optim_groups, lr=self.hparams.lr, betas=(0.9, 0.95), eps=1e-8
        )

        # cosine schedule w/ warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            progress = (step - self.hparams.warmup_steps) / max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}



import wandb
wandb.login()  # will open a prompt the first time

seed_everything(1337, workers=True)

BLOCK_SIZE  = 256
BATCH_SIZE  = 32
MAX_STEPS   = 30_000
PRECISION   = "bf16-mixed" if torch.cuda.is_available() else "32-true"

dm = OWTTextDataModule(
    hf_name="Ankursingh/openwebtext_10K",
    text_col="text",
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=2,
)

dm.prepare_data()
dm.setup()

model = LitGPT(
    vocab_size=dm.vocab_size,
    sequence_length=BLOCK_SIZE,
    n_layer=4, n_head=4, n_embd=256,
    dropout=0.0, bias=False,
    lr=3e-4, weight_decay=0.01,
    warmup_steps=100, max_steps=MAX_STEPS,
)

wandb_logger = loggers.WandbLogger(
    project="tiny-gpt-smoke", 
    name="owt10k-block256",
    dir="wandb/owt-gpt2")
wandb_logger.watch(model, log="all", log_freq=100)

gen_cb = SampleGenerationCallback(
    prompts=("The meaning of life", "In a distant future", "To be or not to be"),
    max_new_tokens=60,
    temperature=0.9,
    top_k=50,
    every_n_steps=None,   # or an int like 500 to log during training too
)

trainer = L.Trainer(
    max_steps=MAX_STEPS,
    log_every_n_steps=10,
    val_check_interval=200,
    gradient_clip_val=1.0,
    precision=PRECISION,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=[6, 7],                    # set to 2/3/4 for multi-GPU
    strategy="ddp" if torch.cuda.device_count() > 1 else None,
    logger=wandb_logger,
    callbacks=[callbacks.LearningRateMonitor(logging_interval="step"), gen_cb],
    enable_checkpointing=False,
)

trainer.fit(model, dm)



# # ---------------------
# # Train entry
# # ---------------------
# def main():
#     seed_everything(1337, workers=True)

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run_name", type=str, default="owt-tiny-base")
#     ap.add_argument("--project", type=str, default="SANIA")
#     ap.add_argument("--max_steps", type=int, default=2000)
#     ap.add_argument("--batch_size", type=int, default=32)
#     ap.add_argument("--seq_len", type=int, default=256)
#     ap.add_argument("--n_layer", type=int, default=4)
#     ap.add_argument("--n_head", type=int, default=4)
#     ap.add_argument("--n_embd", type=int, default=256)
#     ap.add_argument("--dropout", type=float, default=0.0)
#     ap.add_argument("--bias", action="store_true")
#     ap.add_argument("--precision", type=str, default="bf16-mixed")
#     args = ap.parse_args()

#     # Data
#     dm = OWTDataModule(block_size=args.seq_len, batch_size=args.batch_size)
#     dm.setup()

#     # Model config must match your GPTBase expectations
#     model_cfg = ModelCfg(
#         vocab_size=dm.vocab_size,
#         sequence_length=args.seq_len,
#         n_layer=args.n_layer,
#         n_head=args.n_head,
#         n_embd=args.n_embd,
#         dropout=args.dropout,
#         bias=args.bias,
#     )
#     train_cfg = TrainCfg(
#         lr=3e-4,
#         weight_decay=0.01,
#         max_steps=args.max_steps,
#         warmup_steps=100,
#         grad_clip=1.0,
#         precision=args.precision,
#     )
    
#     model = LitGPT(model_cfg, train_cfg)

#     # W&B
#     wandb_logger = loggers.WandbLogger(project=args.project, name=args.run_name, log_model=False)
#     wandb_logger.experiment.config.update({
#         "model_cfg": vars(model_cfg),
#         "train_cfg": vars(train_cfg),
#         "vocab_size": dm.vocab_size,
#     })
#     wandb_logger.watch(model, log="all", log_freq=100)

#     # Callbacks
#     # lr_cb = LearningRateMonitor(logging_interval="step")

#     trainer = L.Trainer(
#         max_steps=train_cfg.max_steps,
#         gradient_clip_val=train_cfg.grad_clip,
#         precision=train_cfg.precision,
#         log_every_n_steps=10,
#         val_check_interval=200,
#         logger=wandb_logger,
#         # callbacks=[lr_cb],
#         enable_checkpointing=False,  # keep runs light; enable if you need checkpoints
#         accumulate_grad_batches=1,
#     )
#     trainer.fit(model, dm)

# if __name__ == "__main__":
#     main()