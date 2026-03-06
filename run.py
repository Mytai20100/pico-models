#!/usr/bin/env python3
"""
pico-model interactive test runner
Usage:
  python run.py llm   --ckpt checkpoints/llm/ckpt_final.pt --tok checkpoints/llm/tokenizer.json
  python run.py img   --ckpt checkpoints/vae/ckpt_vae_final.pt --out samples/
  python run.py tools --ckpt checkpoints/llm/ckpt_final.pt --tok checkpoints/llm/tokenizer.json
"""
import os
import sys
import json
import argparse
import torch

from pico.arch import PicoConfig, PicoLLM, PicoVAE
from pico.tokenizer import CharTokenizer
from pico.tools import default_tools, ToolRegistry


def load_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_llm(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = PicoConfig(**ckpt["config"])
    model = PicoLLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def load_tok(tok_path: str) -> CharTokenizer:
    return CharTokenizer.load(tok_path)


def load_vae(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = PicoConfig(**ckpt["config"])
    model = PicoVAE(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def run_llm(args):
    device = load_device()
    print(f"Loading LLM from {args.ckpt} on {device}")
    model, cfg = load_llm(args.ckpt, device)
    tok = load_tok(args.tok)

    params = model.num_params / 1e6
    print(f"Model: {params:.2f}M params | vocab={cfg.vocab_size} | dim={cfg.dim} | layers={cfg.n_layers}")
    print("Type your prompt. Empty line to quit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break

        ids = tok.encode(prompt, bos=True, eos=False)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)

        output = model.generate(
            tokens,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            stop_ids=[tok.EOS],
        )
        generated = output[0, len(ids):].tolist()
        print(tok.decode(generated))
        print()


def run_img(args):
    device = load_device()
    print(f"Loading VAE from {args.ckpt} on {device}")
    model, cfg = load_vae(args.ckpt, device)

    os.makedirs(args.out, exist_ok=True)
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
    except ImportError:
        print("pip install Pillow torchvision")
        sys.exit(1)

    for i in range(args.n):
        img_tensor = model.sample(1, device=device)[0]
        img = TF.to_pil_image(img_tensor.cpu().clamp(0, 1))
        path = os.path.join(args.out, f"sample_{i:04d}.png")
        img.save(path)
        print(f"Saved {path}")

    if args.reconstruct and os.path.isfile(args.reconstruct):
        from PIL import Image
        import torchvision.transforms as T
        src = Image.open(args.reconstruct).convert("RGB")
        tfm = T.Compose([T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()])
        x = tfm(src).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _, _ = model(x)
        out_img = TF.to_pil_image(recon[0].cpu().clamp(0, 1))
        out_img.save(os.path.join(args.out, "reconstruction.png"))
        print("Saved reconstruction.png")


def run_tools(args):
    device = load_device()
    print(f"Loading LLM from {args.ckpt} on {device}")
    model, cfg = load_llm(args.ckpt, device)
    tok = load_tok(args.tok)

    reg = default_tools()

    def model_fn(prompt: str, device=device) -> str:
        ids = tok.encode(prompt, bos=True, eos=False)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(tokens, max_new_tokens=256, temperature=0.7, top_p=0.9, stop_ids=[tok.EOS])
        generated = output[0, len(ids):].tolist()
        return tok.decode(generated)

    print("Tool-augmented mode. Available tools:", [t["name"] for t in reg.schema()])
    print("Type your query. Empty to quit.\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        result = reg.run_agent_loop(model_fn, tok, query, device=device)
        print(result)
        print()


def run_benchmark(args):
    device = load_device()
    model, cfg = load_llm(args.ckpt, device)

    params = model.num_params / 1e6
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"{'='*40}")
    print(f"  pico-model benchmark")
    print(f"{'='*40}")
    print(f"  Parameters : {params:.2f}M")
    print(f"  Weight RAM : {mem_mb:.1f} MB")
    print(f"  dim={cfg.dim} | layers={cfg.n_layers} | heads={cfg.n_heads} | kv_heads={cfg.n_kv_heads}")
    print(f"  vocab={cfg.vocab_size} | max_seq={cfg.max_seq_len}")

    if args.tok:
        import time
        tok = load_tok(args.tok)
        prompt = "Hello, I am a tiny language model"
        ids = tok.encode(prompt, bos=True, eos=False)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(tokens, max_new_tokens=50, temperature=1.0)
        elapsed = time.time() - t0
        n_new = out.shape[1] - tokens.shape[1]
        print(f"  Throughput : {n_new/elapsed:.1f} tok/s ({device})")
    print(f"{'='*40}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    lm = sub.add_parser("llm", help="Interactive LLM chat")
    lm.add_argument("--ckpt", required=True)
    lm.add_argument("--tok", required=True)
    lm.add_argument("--max-tokens", type=int, default=200)
    lm.add_argument("--temp", type=float, default=0.8)
    lm.add_argument("--top-p", type=float, default=0.9)

    im = sub.add_parser("img", help="Generate / reconstruct images")
    im.add_argument("--ckpt", required=True)
    im.add_argument("--out", default="samples")
    im.add_argument("-n", type=int, default=4)
    im.add_argument("--reconstruct", default="", help="path to image to reconstruct")

    tl = sub.add_parser("tools", help="LLM with tool use")
    tl.add_argument("--ckpt", required=True)
    tl.add_argument("--tok", required=True)
    tl.add_argument("--max-tokens", type=int, default=256)

    bm = sub.add_parser("bench", help="Benchmark model size + speed")
    bm.add_argument("--ckpt", required=True)
    bm.add_argument("--tok", default="")

    args = p.parse_args()

    if args.cmd == "llm":
        run_llm(args)
    elif args.cmd == "img":
        run_img(args)
    elif args.cmd == "tools":
        run_tools(args)
    elif args.cmd == "bench":
        run_benchmark(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
