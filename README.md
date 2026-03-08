# pico-models

A model supporting LLM text generation, image generation (VAE), and tool use.

**Architecture:** GQA Transformer (RoPE + RMSNorm + SwiGLU) for LLM · Patch-based VAE for images  
**Size:** ~1–10M parameters · ~4–40 MB RAM

---

## Install

```bash
pip install torch torchvision Pillow sentencepiece
```

---

## Training

### LLM — plain text

```bash
python train.py train-llm \
  --data data/corpus.txt \
  --out  checkpoints/llm \
  --dim 256 --layers 6 --heads 8 --kv-heads 2 \
  --seq-len 256 --batch 16 --steps 5000
```

### LLM — chat pairs (JSONL with `prompt`/`response` keys)

```bash
python train.py train-llm \
  --data data/chat.jsonl \
  --out  checkpoints/llm \
  --dim 256 --layers 6 --steps 10000
```

### Image generation (VAE)

```bash
python train.py train-img \
  --data  data/images/ \
  --out   checkpoints/vae \
  --img-size 128 --patch-size 16 \
  --dim 256 --layers 4 --batch 8 --steps 5000
```

Flags available for all commands: `--bf16` (AMP on GPU), `--device cuda|cpu|mps`, `--lr`, `--accum`.

---

## Run / Test

```bash
# interactive chat
python run.py llm   --ckpt checkpoints/llm/ckpt_final.pt  --tok checkpoints/llm/tokenizer.json

# generate images
python run.py img   --ckpt checkpoints/vae/ckpt_vae_final.pt --out samples/ -n 8

# reconstruct an image
python run.py img   --ckpt checkpoints/vae/ckpt_vae_final.pt --out samples/ --reconstruct photo.png

# tool-augmented LLM (calculator, clock, echo)
python run.py tools --ckpt checkpoints/llm/ckpt_final.pt  --tok checkpoints/llm/tokenizer.json

# benchmark (params + throughput)
python run.py bench --ckpt checkpoints/llm/ckpt_final.pt  --tok checkpoints/llm/tokenizer.json
```

---

## Custom Tools

```python
from pico.tools import ToolRegistry

reg = ToolRegistry()
reg.register("search", "Search the web", lambda query: {"result": "..."}, {"query": "string"})
```

---
# New version v0.2
Add vision and thinking =) 
## Presets

| Preset | dim | layers | heads | kv_heads | ~Params |
|--------|-----|--------|-------|----------|---------|
| pico   | 128 | 4      | 4     | 1        | ~0.5M   |
| small  | 256 | 6      | 8     | 2        | ~3M     |
| base   | 512 | 8      | 8     | 2        | ~15M    |