#!/usr/bin/env python3
"""
run.py  –  Interactive inference & benchmarking (v2)

Commands:
  python run.py llm   --ckpt checkpoints/llm/ckpt_final.pt --tok checkpoints/llm/tokenizer.json
  python run.py vlm   --ckpt checkpoints/vlm/ckpt_vlm_final.pt --tok ... --image photo.jpg
  python run.py img   --ckpt checkpoints/vae/ckpt_vae_final.pt --out samples/
  python run.py tools --ckpt checkpoints/llm/ckpt_final.pt --tok ...
  python run.py bench --ckpt checkpoints/llm/ckpt_final.pt --tok ...
"""
import os, sys, json, argparse
import torch

from pico.arch      import PicoConfig, PicoLLM, PicoVLM, PicoVAE
from pico.tokenizer import CharTokenizer
from pico.tools     import default_tools
from pico.trainer   import detect_device


def load_llm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = PicoConfig(**ckpt["config"]); m = PicoLLM(cfg)
    m.load_state_dict(ckpt["model"]); return m.to(device).eval(), cfg

def load_vlm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = PicoConfig(**ckpt["config"]); m = PicoVLM(cfg)
    m.load_state_dict(ckpt["model"]); return m.to(device).eval(), cfg

def load_vae(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = PicoConfig(**ckpt["config"]); m = PicoVAE(cfg)
    m.load_state_dict(ckpt["model"]); return m.to(device).eval(), cfg

def load_tok(path): return CharTokenizer.load(path)


# ── LLM ──
def run_llm(args):
    device = detect_device()
    model, cfg = load_llm(args.ckpt, device)
    tok = load_tok(args.tok)

    # Detect if this checkpoint was trained with chat format
    # (has <|user|> and <|assistant|> in vocab)
    is_chat = (tok._ch2id.get("<|user|>") is not None
               and tok._ch2id.get("<|assistant|>") is not None)

    print(f"PicoLLM {model.num_params/1e6:.2f}M | dim={cfg.dim} | layers={cfg.n_layers}")
    print(f"Mode: {'chat' if is_chat else 'text'} | thinking: {cfg.use_thinking}")
    print("Type prompt, empty to quit.\n")

    while True:
        try: prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not prompt: break

        # Wrap in chat format so model knows to enter assistant mode
        if is_chat:
            formatted = f"<|user|>{prompt}<|assistant|>"
        else:
            formatted = prompt

        ids    = tok.encode(formatted, bos=True, eos=False)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        out    = model.generate(tokens, max_new_tokens=args.max_tokens,
                                temperature=args.temp, top_p=args.top_p,
                                stop_ids=[tok.EOS])
        generated = out[0, len(ids):].tolist()

        if cfg.use_thinking and hasattr(tok, "decode_with_thinking"):
            answer, thinking = tok.decode_with_thinking(generated)
            if args.show_thinking and thinking:
                print(f"[Thinking] {thinking}")
            print(answer)
        else:
            print(tok.decode(generated))
        print()


# ── VLM ──
def run_vlm(args):
    device = detect_device()
    model, cfg = load_vlm(args.ckpt, device)
    tok = load_tok(args.tok)
    print(f"PicoVLM {model.num_params/1e6:.2f}M | image support ✓")

    image_tensor = None
    if args.image and os.path.exists(args.image):
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(args.image).convert("RGB")
            tfm = T.Compose([T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor(),
                              T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
            image_tensor = tfm(img).unsqueeze(0).to(device)
            print(f"  Image loaded: {args.image}")
        except Exception as e:
            print(f"  Image load failed: {e}")

    print("Type prompt, empty to quit.\n")
    while True:
        try: prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not prompt: break
        ids    = tok.encode(f"<|user|>{prompt}<|assistant|>", bos=True, eos=False)
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        out    = model.generate(tokens, images=image_tensor,
                                max_new_tokens=args.max_tokens,
                                temperature=args.temp, top_p=args.top_p,
                                stop_ids=[tok.EOS])
        print(tok.decode(out[0, len(ids):].tolist()))
        print()


# ── VAE image generation ──
def run_img(args):
    device = detect_device()
    model, cfg = load_vae(args.ckpt, device)
    os.makedirs(args.out, exist_ok=True)
    try:
        from PIL import Image; import torchvision.transforms.functional as TF
    except ImportError:
        print("pip install Pillow torchvision"); sys.exit(1)

    for i in range(args.n):
        img_t = model.sample(1, device=device)[0]
        TF.to_pil_image(img_t.cpu().clamp(0,1)).save(
            os.path.join(args.out, f"sample_{i:04d}.png"))
        print(f"  saved sample_{i:04d}.png")

    if args.reconstruct and os.path.isfile(args.reconstruct):
        from PIL import Image; import torchvision.transforms as T
        img = Image.open(args.reconstruct).convert("RGB")
        tfm = T.Compose([T.Resize((cfg.img_size,cfg.img_size)), T.ToTensor()])
        x   = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad(): recon, _, _, _ = model(x)
        TF.to_pil_image(recon[0].cpu().clamp(0,1)).save(
            os.path.join(args.out, "reconstruction.png"))
        print("  saved reconstruction.png")


# ── Tool use ──
def run_tools(args):
    device = detect_device()
    model, cfg = load_llm(args.ckpt, device)
    tok = load_tok(args.tok)
    reg = default_tools()

    def model_fn(prompt, device=device):
        ids = tok.encode(prompt, bos=True, eos=False)
        t   = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(t, max_new_tokens=256, temperature=0.7, stop_ids=[tok.EOS])
        return tok.decode(out[0, len(ids):].tolist())

    print("Tool-augmented mode:", [t["name"] for t in reg.schema()])
    while True:
        try: q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not q: break
        print(reg.run_agent_loop(model_fn, tok, q, device=device)); print()


# ── Benchmark ──
def run_benchmark(args):
    import time
    device = detect_device()
    model, cfg = load_llm(args.ckpt, device)
    params = model.num_params / 1e6
    mem_mb = sum(p.numel()*p.element_size() for p in model.parameters()) / 1e6
    print(f"{'='*48}")
    print(f"  pico-model benchmark  (v2)")
    print(f"{'='*48}")
    print(f"  Params     : {params:.2f}M")
    print(f"  Weight RAM : {mem_mb:.1f} MB")
    print(f"  dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} kv={cfg.n_kv_heads}")
    print(f"  vocab={cfg.vocab_size} max_seq={cfg.max_seq_len}")
    print(f"  Device: {device}")
    if args.tok:
        tok   = load_tok(args.tok)
        ids   = tok.encode("Hello, I am a small language model.", bos=True, eos=False)
        toks  = torch.tensor([ids], dtype=torch.long, device=device)
        # warmup
        with torch.no_grad(): _ = model.generate(toks, max_new_tokens=10)
        t0 = time.time()
        with torch.no_grad(): out = model.generate(toks, max_new_tokens=100)
        elapsed = time.time() - t0
        n = out.shape[1] - toks.shape[1]
        print(f"  Throughput : {n/elapsed:.1f} tok/s")
    print(f"{'='*48}")


# ─────────────────────────── Main ───────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    lm = sub.add_parser("llm");   lm.add_argument("--ckpt",required=True); lm.add_argument("--tok",required=True)
    lm.add_argument("--max-tokens",type=int,default=200); lm.add_argument("--temp",type=float,default=0.8)
    lm.add_argument("--top-p",type=float,default=0.9); lm.add_argument("--show-thinking",action="store_true")

    vm = sub.add_parser("vlm");   vm.add_argument("--ckpt",required=True); vm.add_argument("--tok",required=True)
    vm.add_argument("--image",default=""); vm.add_argument("--max-tokens",type=int,default=200)
    vm.add_argument("--temp",type=float,default=0.8); vm.add_argument("--top-p",type=float,default=0.9)

    im = sub.add_parser("img");   im.add_argument("--ckpt",required=True); im.add_argument("--out",default="samples")
    im.add_argument("-n",type=int,default=4); im.add_argument("--reconstruct",default="")

    tl = sub.add_parser("tools"); tl.add_argument("--ckpt",required=True); tl.add_argument("--tok",required=True)

    bm = sub.add_parser("bench"); bm.add_argument("--ckpt",required=True); bm.add_argument("--tok",default="")

    args = p.parse_args()
    dispatch = {"llm":run_llm,"vlm":run_vlm,"img":run_img,"tools":run_tools,"bench":run_benchmark}
    fn = dispatch.get(args.cmd)
    if fn: fn(args)
    else:  p.print_help()

if __name__ == "__main__":
    main()
