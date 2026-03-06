import os
import time
import math
import json
from dataclasses import dataclass, asdict
from typing import Optional
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from pico.arch import PicoConfig, PicoLLM, PicoVAE


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints"
    max_steps: int = 5000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 16
    grad_accum: int = 1
    fp16: bool = False
    bf16: bool = False
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500
    seed: int = 42


def get_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def save_checkpoint(model, optimizer, step, loss, out_dir, tag="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{tag}.pt")
    torch.save({
        "step": step,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(model.cfg),
    }, path)
    return path


def load_checkpoint(path: str, model_cls, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    cfg_cls = PicoConfig
    cfg = cfg_cls(**cfg_dict)
    model = model_cls(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt.get("step", 0)


class Trainer:
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg = train_cfg
        self.device = device

        torch.manual_seed(train_cfg.seed)

        no_decay = {"bias", "norm", "embed"}
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": train_cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
        ]
        self.opt = torch.optim.AdamW(params, lr=train_cfg.lr, betas=(0.9, 0.95), eps=1e-8)

        dtype = torch.bfloat16 if train_cfg.bf16 else (torch.float16 if train_cfg.fp16 else None)
        self.amp = dtype is not None
        self.dtype = dtype
        self.scaler = GradScaler("cuda") if (self.amp and not train_cfg.bf16) else None
        self.step = 0

        # torch.compile for ~20-40% speedup on modern GPUs
        if device == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("torch.compile enabled ✓")
            except Exception:
                pass

    def train_llm(self, loader_fn, total_steps: Optional[int] = None):
        steps = total_steps or self.cfg.max_steps
        loader = loader_fn()
        it = iter(loader)

        self.model.train()
        t0 = time.time()
        running_loss = 0.0

        while self.step < steps:
            lr = get_lr(self.step, self.cfg, steps)
            for pg in self.opt.param_groups:
                pg["lr"] = lr

            self.opt.zero_grad(set_to_none=True)
            acc_loss = 0.0

            for _ in range(self.cfg.grad_accum):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, y = next(it)

                x, y = x.to(self.device), y.to(self.device)

                if self.amp:
                    with autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
                        _, loss, _ = self.model(x, y)
                else:
                    _, loss, _ = self.model(x, y)

                loss = loss / self.cfg.grad_accum
                acc_loss += loss.item()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self.scaler:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()

            running_loss += acc_loss
            self.step += 1

            if self.step % self.cfg.log_every == 0:
                avg = running_loss / self.cfg.log_every
                elapsed = time.time() - t0
                print(f"step={self.step:6d} | loss={avg:.4f} | lr={lr:.2e} | {elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

            if self.step % self.cfg.save_every == 0:
                save_checkpoint(self.model, self.opt, self.step, acc_loss, self.cfg.out_dir)
                print(f"  saved checkpoint at step {self.step}")

        save_checkpoint(self.model, self.opt, self.step, acc_loss, self.cfg.out_dir, tag="final")
        print("Training complete.")

    def train_vae(self, loader_fn, total_steps: Optional[int] = None):
        steps = total_steps or self.cfg.max_steps
        loader = loader_fn()
        it = iter(loader)

        self.model.train()
        t0 = time.time()
        running_loss = 0.0

        while self.step < steps:
            lr = get_lr(self.step, self.cfg, steps)
            for pg in self.opt.param_groups:
                pg["lr"] = lr

            try:
                x = next(it)
            except StopIteration:
                it = iter(loader)
                x = next(it)

            x = x.to(self.device)
            self.opt.zero_grad(set_to_none=True)

            if self.amp:
                with autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
                    _, loss, _, _ = self.model(x)
            else:
                _, loss, _, _ = self.model(x)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()

            running_loss += loss.item()
            self.step += 1

            if self.step % self.cfg.log_every == 0:
                avg = running_loss / self.cfg.log_every
                elapsed = time.time() - t0
                print(f"step={self.step:6d} | vae_loss={avg:.4f} | lr={lr:.2e} | {elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

            if self.step % self.cfg.save_every == 0:
                save_checkpoint(self.model, self.opt, self.step, loss.item(), self.cfg.out_dir, tag=f"vae_{self.step}")

        save_checkpoint(self.model, self.opt, self.step, loss.item(), self.cfg.out_dir, tag="vae_final")
        print("VAE training complete.")
