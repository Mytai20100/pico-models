#!/usr/bin/env python3
"""
train.py  –  Pico Model Training CLI (v2.1)

Optimised for NVIDIA T4 (15 GB) — but auto-adapts to any GPU.
"""
import os, sys, json, argparse
import torch

from pico.arch      import PicoConfig, PicoLLM, PicoVLM, PicoVAE
from pico.trainer   import Trainer, TrainConfig, detect_device
from pico.tokenizer import get_tokenizer
from pico.data      import (
    load_txt, load_jsonl,
    make_text_loader, make_chat_loader, make_thinking_loader,
    make_image_loader, make_vlm_loader,
)


def _add_common(p):
    p.add_argument("--dim",       type=int,   default=256)
    p.add_argument("--layers",    type=int,   default=6)
    p.add_argument("--heads",     type=int,   default=8)
    p.add_argument("--kv-heads",  type=int,   default=2)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--steps",     type=int,   default=5000)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--accum",     type=int,   default=1,
                   help="Gradient accumulation steps (increase to simulate larger batch without more VRAM)")
    p.add_argument("--fp16",      action="store_true", help="Force fp16 (auto on T4/V100)")
    p.add_argument("--bf16",      action="store_true", help="Force bf16 (auto on Ampere+)")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Gradient checkpointing — saves ~40%% VRAM, ~15%% slower")
    p.add_argument("--device",    default="")
    p.add_argument("--out",       default="checkpoints")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every",type=int, default=1000)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    # ── train-llm ──
    t = sub.add_parser("train-llm", help="Train text LLM")
    t.add_argument("--data",      required=True)
    t.add_argument("--tokenizer", default="char", choices=["char","sp"])
    t.add_argument("--vocab-size",type=int,   default=1024)
    t.add_argument("--seq-len",   type=int,   default=256)
    t.add_argument("--thinking",  action="store_true")
    _add_common(t)

    # ── train-vlm ──
    v = sub.add_parser("train-vlm", help="Train Vision-Language Model")
    v.add_argument("--data",      required=True,
                   help="JSONL: {image, prompt, response[, thinking]}")
    v.add_argument("--tokenizer", default="char", choices=["char","sp"])
    v.add_argument("--vocab-size",type=int,   default=1024)
    v.add_argument("--seq-len",   type=int,   default=256)
    v.add_argument("--img-size",  type=int,   default=128)
    v.add_argument("--patch-size",type=int,   default=16)
    v.add_argument("--thinking",  action="store_true")
    _add_common(v)

    # ── train-img (VAE) ──
    i = sub.add_parser("train-img", help="Train image VAE")
    i.add_argument("--data",       required=True, help="Folder of images")
    i.add_argument("--img-size",   type=int,   default=128)
    i.add_argument("--patch-size", type=int,   default=16)
    i.add_argument("--latent-dim", type=int,   default=64)
    _add_common(i)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────

def _make_tcfg(args, out_dir, device) -> TrainConfig:
    return TrainConfig(
        out_dir    = out_dir,
        max_steps  = args.steps,
        lr         = args.lr,
        batch_size = args.batch,
        grad_accum = args.accum,
        fp16       = args.fp16,
        bf16       = args.bf16 and device != "cpu",
        grad_ckpt  = args.grad_ckpt,
        log_every  = args.log_every,
        save_every = args.save_every,
        use_thinking = getattr(args, "thinking", False),
    )


def cmd_train_llm(args):
    device = detect_device(args.device)
    out    = args.out if args.out != "checkpoints" else "checkpoints/llm"

    ext     = os.path.splitext(args.data)[1].lower()
    is_chat = (ext == ".jsonl")
    tok     = get_tokenizer(args.tokenizer, vocab_size=args.vocab_size)

    if is_chat:
        pairs = load_jsonl(args.data)
        texts = []
        for p in pairs:
            th = p.get("thinking",""); r = p.get("response",p.get("output",""))
            pr = p.get("prompt",p.get("input",""))
            texts.append(f"<|user|>{pr}<|assistant|><think>{th}</think>{r}" if th
                         else f"<|user|>{pr}<|assistant|>{r}")
        tok.train(texts)
    else:
        tok.train([load_txt(args.data)])

    os.makedirs(out, exist_ok=True)
    tok.save(os.path.join(out, "tokenizer.json"))

    cfg = PicoConfig(
        vocab_size=len(tok), dim=args.dim,
        n_layers=args.layers, n_heads=args.heads, n_kv_heads=args.kv_heads,
        max_seq_len=args.seq_len, model_type="llm",
        use_thinking=args.thinking,
        think_start_id=getattr(tok, "THINK_START", -1),
        think_end_id  =getattr(tok, "THINK_END",   -1),
    )
    model = PicoLLM(cfg)
    print(f"  PicoLLM  {model.num_params/1e6:.2f}M params | vocab={cfg.vocab_size}")

    if is_chat and args.thinking:
        loader_fn = lambda: make_thinking_loader(pairs, tok, args.seq_len, args.batch)
    elif is_chat:
        loader_fn = lambda: make_chat_loader(pairs, tok, args.seq_len, args.batch)
    else:
        tokens    = tok.encode(load_txt(args.data), bos=False, eos=False)
        loader_fn = lambda: make_text_loader(tokens, args.seq_len, args.batch)

    Trainer(model, _make_tcfg(args, out, device), device).train_llm(loader_fn, args.steps)


def cmd_train_vlm(args):
    device = detect_device(args.device)
    out    = args.out if args.out != "checkpoints" else "checkpoints/vlm"

    pairs  = load_jsonl(args.data)
    tok    = get_tokenizer(args.tokenizer, vocab_size=args.vocab_size)
    texts  = []
    for p in pairs:
        th = p.get("thinking",""); r = p.get("response",""); pr = p.get("prompt","")
        texts.append(f"<|user|>{pr}<|assistant|><think>{th}</think>{r}" if th
                     else f"<|user|>{pr}<|assistant|>{r}")
    tok.train(texts)
    os.makedirs(out, exist_ok=True)
    tok.save(os.path.join(out, "tokenizer.json"))

    cfg = PicoConfig(
        vocab_size=len(tok), dim=args.dim,
        n_layers=args.layers, n_heads=args.heads, n_kv_heads=args.kv_heads,
        max_seq_len=args.seq_len, model_type="vlm",
        img_size=args.img_size, img_patch_size=args.patch_size,
        use_thinking=args.thinking,
        think_start_id=getattr(tok, "THINK_START", -1),
        think_end_id  =getattr(tok, "THINK_END",   -1),
    )
    model = PicoVLM(cfg)
    print(f"  PicoVLM  {model.num_params/1e6:.2f}M params | vocab={cfg.vocab_size}")
    loader_fn = lambda: make_vlm_loader(args.data, tok, args.seq_len,
                                        args.img_size, args.batch)
    Trainer(model, _make_tcfg(args, out, device), device).train_vlm(loader_fn, args.steps)


def cmd_train_img(args):
    device = detect_device(args.device)
    out    = args.out if args.out != "checkpoints" else "checkpoints/vae"

    cfg = PicoConfig(
        dim=args.dim, n_layers=args.layers, n_heads=args.heads,
        n_kv_heads=min(2, args.heads),
        img_size=args.img_size, img_patch_size=args.patch_size,
        latent_dim=args.latent_dim, model_type="vae",
    )
    model = PicoVAE(cfg)
    print(f"  PicoVAE  {model.num_params/1e6:.2f}M params")
    loader_fn = lambda: make_image_loader(args.data, args.img_size, args.batch)
    Trainer(model, _make_tcfg(args, out, device), device).train_vae(loader_fn, args.steps)


def main():
    args = parse_args()
    if   args.cmd == "train-llm": cmd_train_llm(args)
    elif args.cmd == "train-vlm": cmd_train_vlm(args)
    elif args.cmd == "train-img": cmd_train_img(args)
    else:
        print("Usage: python train.py {train-llm|train-vlm|train-img} --help")
        sys.exit(1)

if __name__ == "__main__":
    main()
