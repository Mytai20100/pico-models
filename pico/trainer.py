"""
pico/trainer.py  –  Training Engine (v2.1 — T4/Colab optimised)

GPU Speed:
  • cudnn.benchmark + TF32 (Ampere+)
  • torch.compile:  max-autotune (Ampere+)  /  reduce-overhead (Turing/T4)
  • Auto fp16 on T4  (cc 7.5, no native BF16 tensor cores)
  • GradScaler for fp16 stability
  • Gradient Checkpointing  (saves ~40% VRAM, ~15% slower)
  • DataLoader: persistent_workers + prefetch_factor=4 + pin_memory
  • Multi-GPU: DDP (torchrun) or DataParallel fallback
  • tokens/sec throughput logging

NPU/TPU:
  • Google TPU  (torch_xla)
  • Intel XPU   (intel_extension_for_pytorch)
  • Apple MPS
  • DirectML
"""
from __future__ import annotations

import os, time, math, json
from dataclasses import dataclass, asdict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from pico.arch import PicoConfig, PicoLLM, PicoVLM, PicoVAE


# Device helpers 

def detect_device(requested: str = "") -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm; _ = xm.xla_device(); return "xla"
    except Exception: pass
    try:
        import intel_extension_for_pytorch as ipex  # noqa
        if torch.xpu.is_available(): return "xpu"
    except Exception: pass
    if torch.backends.mps.is_available():
        return "mps"
    try:
        import torch_directml; return torch_directml.device()  # noqa
    except Exception: pass
    return "cpu"


def _device_type(device: str) -> str:
    for prefix in ("xla", "xpu", "cuda", "mps"):
        if device.startswith(prefix): return prefix
    return "cpu"


def _gpu_info() -> dict:
    """Return compute capability and VRAM (GB) for current CUDA device."""
    if not torch.cuda.is_available():
        return {"cc": (0, 0), "vram_gb": 0}
    cc     = torch.cuda.get_device_capability()
    vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {"cc": cc, "vram_gb": vram}


def _best_dtype(bf16_flag: bool, fp16_flag: bool) -> Optional[torch.dtype]:
    """
    Pick the best precision automatically:
    - If user forces bf16/fp16, respect that.
    - T4 (cc 7.5) → fp16  (BF16 has no tensor-core acceleration on Turing)
    - Ampere+ (cc 8.x) → bf16  (native BF16 tensor cores, more stable)
    - Older / CPU → None (fp32)
    """
    if bf16_flag: return torch.bfloat16
    if fp16_flag: return torch.float16
    if not torch.cuda.is_available(): return None
    cc = torch.cuda.get_device_capability()
    if cc[0] >= 8:   return torch.bfloat16   # Ampere+
    if cc[0] >= 7:   return torch.float16    # Volta / Turing (T4, V100)
    return None


def _setup_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:                              # Ampere+ only
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True


def _maybe_compile(model: nn.Module, device: str) -> nn.Module:
    if device not in ("cuda", "xpu"):
        return model
    try:
        cc = torch.cuda.get_device_capability() if device == "cuda" else (0, 0)
        if cc[0] < 8:
            # Turing/Volta (T4, V100): skip compile entirely.
            # "reduce-overhead" uses CUDA Graphs which break training loops
            # (CUDAGraph tensor overwrite error). "max-autotune" needs more SMs.
            # Plain eager + fp16 + cudnn.benchmark is already optimal on T4.
            print(f"  torch.compile skipped (cc {cc[0]}.{cc[1]} < 8.0 — not beneficial for training)")
            return model
        # Ampere+ only: use default mode (no CUDA Graphs) to stay safe in training
        model = torch.compile(model, mode="default")
        print("  torch.compile (default) ✓")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")
    return model


def _maybe_multi_gpu(model: nn.Module) -> nn.Module:
    if not torch.cuda.is_available(): return model
    if "LOCAL_RANK" in os.environ:
        lr = int(os.environ["LOCAL_RANK"])
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(lr)
        model = model.to(lr)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[lr], find_unused_parameters=False)
        print(f"  DDP rank {lr} ✓")
        return model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel ×{torch.cuda.device_count()} ✓")
    return model


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable activation checkpointing on PicoBlock layers.
    Saves ~40% VRAM at cost of ~15% extra compute (recomputes activations on backward).
    Critical for fitting VLM on T4 15GB with larger batches.
    """
    from torch.utils.checkpoint import checkpoint

    raw = model.module if hasattr(model, "module") else model
    layers = getattr(raw, "layers", None)
    if layers is None: return

    # Monkey-patch each block's forward to use checkpoint
    original_forwards = {}
    for i, blk in enumerate(layers):
        orig = blk.forward
        original_forwards[i] = orig

        def make_ckpt_forward(fn):
            def ckpt_forward(x, freqs_cis, mask=None, kv_cache=None):
                # checkpoint only works when kv_cache is None (training)
                if kv_cache is None:
                    def run(x_, fc_): return fn(x_, fc_, mask, None)
                    x_out, cache = checkpoint(run, x, freqs_cis, use_reentrant=False)
                    return x_out, cache
                return fn(x, freqs_cis, mask, kv_cache)
            return ckpt_forward

        blk.forward = make_ckpt_forward(orig)

    print(f"  Gradient checkpointing enabled on {len(layers)} layers ✓")


# Config 

@dataclass
class TrainConfig:
    out_dir:        str   = "checkpoints"
    max_steps:      int   = 5000
    warmup_steps:   int   = 200
    lr:             float = 3e-4
    min_lr:         float = 3e-5
    weight_decay:   float = 0.1
    grad_clip:      float = 1.0
    batch_size:     int   = 16
    grad_accum:     int   = 1
    fp16:           bool  = False
    bf16:           bool  = False
    log_every:      int   = 100
    save_every:     int   = 1000
    eval_every:     int   = 500
    seed:           int   = 42
    use_thinking:   bool  = False
    grad_ckpt:      bool  = False   # gradient checkpointing (VRAM saver)


def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * p))


#  Checkpoint 

def save_checkpoint(model, optimizer, step, loss, out_dir, tag="latest"):
    os.makedirs(out_dir, exist_ok=True)
    raw  = model.module if hasattr(model, "module") else model
    path = os.path.join(out_dir, f"ckpt_{tag}.pt")
    torch.save({"step": step, "loss": loss,
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(raw.cfg)}, path)
    return path

def load_checkpoint(path, model_cls, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = PicoConfig(**ckpt["config"])
    m    = model_cls(cfg)
    m.load_state_dict(ckpt["model"])
    return m.to(device), ckpt.get("step", 0)


# Trainer 

class Trainer:
    def __init__(self, model: nn.Module, train_cfg: TrainConfig, device: str = "cpu"):
        _setup_gpu()
        torch.manual_seed(train_cfg.seed)

        self.cfg    = train_cfg
        self.device = device

        # Auto-select best precision for this GPU
        self.dtype = _best_dtype(train_cfg.bf16, train_cfg.fp16)
        self.amp   = (self.dtype is not None) and (device not in ("cpu", "mps"))

        info = _gpu_info()
        print(f"\n{'='*55}")
        print(f"  Pico Trainer — device={device}")
        if device == "cuda":
            name = torch.cuda.get_device_name(0)
            print(f"  GPU: {name}  ({info['vram_gb']:.1f} GB)  cc={info['cc']}")
        print(f"  Precision: {self.dtype}  |  AMP: {self.amp}")
        print(f"  Effective batch: {train_cfg.batch_size} × {train_cfg.grad_accum} = "
              f"{train_cfg.batch_size * train_cfg.grad_accum}")
        print(f"{'='*55}\n")

        model = model.to(device)

        if train_cfg.grad_ckpt:
            enable_gradient_checkpointing(model)

        model = _maybe_compile(model, device)
        model = _maybe_multi_gpu(model)
        self.model = model

        # Intel XPU
        if device == "xpu":
            try:
                import intel_extension_for_pytorch as ipex  # noqa
                self.model, _ = ipex.optimize(self.model, dtype=self.dtype or torch.float32)
                print("  Intel XPU optimised ✓")
            except Exception: pass

        # Optimizer — separate weight decay groups
        no_decay = {"bias", "norm", "embed"}
        params = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": train_cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        self.opt = torch.optim.AdamW(params, lr=train_cfg.lr, betas=(0.9, 0.95), eps=1e-8,
                                     fused=(device == "cuda"))   # fused AdamW on CUDA

        # GradScaler for fp16 only
        self.scaler = (GradScaler("cuda")
                       if (self.amp and self.dtype == torch.float16 and device == "cuda")
                       else None)
        self.step = 0; self._tokens_seen = 0; self._t0 = time.time()

    #  internals

    def _set_lr(self, total):
        lr = get_lr(self.step, self.cfg, total)
        for pg in self.opt.param_groups: pg["lr"] = lr
        return lr

    def _autocast(self):
        dt = _device_type(self.device)
        if not self.amp or dt in ("mps", "cpu", "xla"):
            from contextlib import nullcontext; return nullcontext()
        return autocast(device_type=dt, dtype=self.dtype)

    def _backward(self, loss):
        if self.scaler: self.scaler.scale(loss).backward()
        else:           loss.backward()

    def _opt_step(self):
        if self.scaler:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt); self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()

    def _maybe_xla_mark_step(self):
        if self.device == "xla":
            try:
                import torch_xla.core.xla_model as xm; xm.mark_step()  # noqa
            except Exception: pass

    def _vram_str(self):
        if self.device != "cuda": return ""
        used = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return f" | VRAM {used:.1f}/{peak:.1f}GB"

    def _log(self, run_loss, lr):
        avg = run_loss / self.cfg.log_every
        elapsed = time.time() - self._t0
        tps = self._tokens_seen / max(elapsed, 1e-6)
        print(f"step={self.step:6d} | loss={avg:.4f} | lr={lr:.2e} | "
              f"{tps:,.0f} tok/s | {elapsed:.1f}s{self._vram_str()}")
        self._tokens_seen = 0; self._t0 = time.time()

     
    #  train_llm
     
    def train_llm(self, loader_fn: Callable, total_steps: Optional[int] = None):
        steps = total_steps or self.cfg.max_steps
        loader = loader_fn(); it = iter(loader)
        self.model.train(); run_loss = 0.0; self._t0 = time.time()

        while self.step < steps:
            lr = self._set_lr(steps)
            self.opt.zero_grad(set_to_none=True)
            acc_loss = 0.0

            for _ in range(self.cfg.grad_accum):
                try:    batch = next(it)
                except StopIteration: it = iter(loader); batch = next(it)
                x, y   = batch[0], batch[1]
                tmask  = batch[2] if len(batch) > 2 else None
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if tmask is not None:
                    tmask = tmask.to(self.device, non_blocking=True)

                with self._autocast():
                    _, loss, _ = self.model(x, y, thinking_mask=tmask)
                loss = loss / self.cfg.grad_accum
                acc_loss += loss.item()
                self._backward(loss)
                self._tokens_seen += x.numel()

            self._opt_step(); self._maybe_xla_mark_step()
            run_loss += acc_loss; self.step += 1

            if self.step % self.cfg.log_every == 0:
                self._log(run_loss, lr); run_loss = 0.0
            if self.step % self.cfg.save_every == 0:
                p = save_checkpoint(self.model, self.opt, self.step, acc_loss, self.cfg.out_dir)
                print(f"  ✓ saved {p}")

        save_checkpoint(self.model, self.opt, self.step, acc_loss, self.cfg.out_dir, "final")
        print("LLM training complete ✓")

     
    #  train_vae
     
    def train_vae(self, loader_fn: Callable, total_steps: Optional[int] = None):
        steps = total_steps or self.cfg.max_steps
        loader = loader_fn(); it = iter(loader)
        self.model.train(); run_loss = 0.0; self._t0 = time.time()

        while self.step < steps:
            lr = self._set_lr(steps)
            self.opt.zero_grad(set_to_none=True)
            acc_loss = 0.0

            for _ in range(self.cfg.grad_accum):
                try:    x = next(it)
                except StopIteration: it = iter(loader); x = next(it)
                x = x.to(self.device, non_blocking=True)
                with self._autocast():
                    _, loss, _, _ = self.model(x)
                loss = loss / self.cfg.grad_accum
                acc_loss += loss.item()
                self._backward(loss)

            self._opt_step(); self._maybe_xla_mark_step()
            run_loss += acc_loss; self.step += 1

            if self.step % self.cfg.log_every == 0:
                avg = run_loss / self.cfg.log_every
                elapsed = time.time() - self._t0
                print(f"step={self.step:6d} | vae_loss={avg:.4f} | lr={lr:.2e} | "
                      f"{elapsed:.1f}s{self._vram_str()}")
                run_loss = 0.0; self._t0 = time.time()
            if self.step % self.cfg.save_every == 0:
                p = save_checkpoint(self.model, self.opt, self.step, acc_loss,
                                    self.cfg.out_dir, f"vae_{self.step}")
                print(f"  ✓ saved {p}")

        save_checkpoint(self.model, self.opt, self.step, acc_loss,
                        self.cfg.out_dir, "vae_final")
        print("VAE training complete ✓")

     
    #  train_vlm
    def train_vlm(self, loader_fn: Callable, total_steps: Optional[int] = None):
        steps = total_steps or self.cfg.max_steps
        loader = loader_fn(); it = iter(loader)
        self.model.train(); run_loss = 0.0; self._t0 = time.time()

        while self.step < steps:
            lr = self._set_lr(steps)
            self.opt.zero_grad(set_to_none=True)
            acc_loss = 0.0

            for _ in range(self.cfg.grad_accum):
                try:    batch = next(it)
                except StopIteration: it = iter(loader); batch = next(it)
                imgs, x, y, tmask = batch
                imgs  = imgs.to(self.device,  non_blocking=True)
                x     = x.to(self.device,     non_blocking=True)
                y     = y.to(self.device,     non_blocking=True)
                tmask = tmask.to(self.device,  non_blocking=True)

                with self._autocast():
                    _, loss, _ = self.model(x, targets=y, images=imgs, thinking_mask=tmask)
                loss = loss / self.cfg.grad_accum
                acc_loss += loss.item()
                self._backward(loss)
                self._tokens_seen += x.numel()

            self._opt_step(); self._maybe_xla_mark_step()
            run_loss += acc_loss; self.step += 1

            if self.step % self.cfg.log_every == 0:
                self._log(run_loss, lr); run_loss = 0.0
            if self.step % self.cfg.save_every == 0:
                p = save_checkpoint(self.model, self.opt, self.step, acc_loss,
                                    self.cfg.out_dir, f"vlm_{self.step}")
                print(f"  ✓ saved {p}")

        save_checkpoint(self.model, self.opt, self.step, acc_loss,
                        self.cfg.out_dir, "vlm_final")
        print("VLM training complete ✓")
