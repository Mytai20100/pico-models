#!/usr/bin/env python3
import os
import sys
import json
import argparse

import torch

from pico.arch import PicoConfig, PicoLLM, PicoVAE
from pico.trainer import Trainer, TrainConfig
from pico.tokenizer import CharTokenizer, get_tokenizer
from pico.data import (
    load_txt, load_jsonl,
    make_text_loader, make_chat_loader, make_image_loader,
)


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train-llm")
    t.add_argument("--data", required=True, help=".txt or .jsonl file")
    t.add_argument("--out", default="checkpoints/llm")
    t.add_argument("--tokenizer", default="char")
    t.add_argument("--vocab-size", type=int, default=1024)
    t.add_argument("--dim", type=int, default=256)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--kv-heads", type=int, default=2)
    t.add_argument("--seq-len", type=int, default=256)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--device", default="")

    v = sub.add_parser("train-img")
    v.add_argument("--data", required=True, help="folder of images")
    v.add_argument("--out", default="checkpoints/vae")
    v.add_argument("--img-size", type=int, default=128)
    v.add_argument("--patch-size", type=int, default=16)
    v.add_argument("--latent-dim", type=int, default=64)
    v.add_argument("--dim", type=int, default=256)
    v.add_argument("--layers", type=int, default=4)
    v.add_argument("--heads", type=int, default=8)
    v.add_argument("--batch", type=int, default=8)
    v.add_argument("--steps", type=int, default=5000)
    v.add_argument("--lr", type=float, default=1e-3)
    v.add_argument("--bf16", action="store_true")
    v.add_argument("--device", default="")

    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cmd_train_llm(args):
    device = resolve_device(args.device)
    print(f"Device: {device}")

    ext = os.path.splitext(args.data)[1].lower()
    is_chat = ext == ".jsonl"

    tok = get_tokenizer(args.tokenizer, vocab_size=args.vocab_size)

    if is_chat:
        pairs = load_jsonl(args.data)
        texts = [f"User: {p.get('prompt', p.get('input', ''))}\nAssistant: {p.get('response', p.get('output', ''))}" for p in pairs]
        tok.train(texts)
    else:
        raw = load_txt(args.data)
        tok.train([raw])

    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    cfg = PicoConfig(
        vocab_size=len(tok),
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        max_seq_len=args.seq_len,
        model_type="llm",
    )
    model = PicoLLM(cfg)
    params = model.num_params / 1e6
    print(f"Model: {params:.2f}M params | vocab={cfg.vocab_size}")

    tcfg = TrainConfig(
        out_dir=args.out,
        max_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch,
        grad_accum=args.accum,
        bf16=args.bf16 and device != "cpu",
    )

    if is_chat:
        loader_fn = lambda: make_chat_loader(pairs, tok, args.seq_len, args.batch)
    else:
        tokens = tok.encode(load_txt(args.data), bos=False, eos=False)
        loader_fn = lambda: make_text_loader(tokens, args.seq_len, args.batch)

    trainer = Trainer(model, tcfg, device)
    trainer.train_llm(loader_fn, args.steps)


def cmd_train_img(args):
    device = resolve_device(args.device)
    print(f"Device: {device}")

    cfg = PicoConfig(
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=min(2, args.heads),
        img_size=args.img_size,
        img_patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        model_type="vae",
    )
    model = PicoVAE(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"VAE: {params:.2f}M params")

    tcfg = TrainConfig(
        out_dir=args.out,
        max_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch,
        bf16=args.bf16 and device != "cpu",
    )

    loader_fn = lambda: make_image_loader(args.data, args.img_size, args.batch)
    trainer = Trainer(model, tcfg, device)
    trainer.train_vae(loader_fn, args.steps)


def main():
    args = parse_args()
    if args.cmd == "train-llm":
        cmd_train_llm(args)
    elif args.cmd == "train-img":
        cmd_train_img(args)
    else:
        print("Usage: python train.py {train-llm|train-img} --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
