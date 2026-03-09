"""
pico/data.py  –  Dataset & DataLoader utilities (v2)

New in v2:
  • ThinkingDataset  – text/chat with <think> trace support + thinking masks
  • VisionChatDataset – image + text pairs for VLM training
  • Optimised DataLoaders: persistent_workers, prefetch_factor, pin_memory
  • num_workers auto-scaled to CPU count
  • Collation supports variable-length sequences
"""
import os, json, random
from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

# DataLoader best settings 

def _optimal_workers(requested: int = 0) -> int:
    cpu = os.cpu_count() or 4
    return min(requested if requested > 0 else cpu, 8)   # cap at 8 for stability

def _make_loader(ds, batch_size, shuffle, collate_fn=None, num_workers=0,
                 drop_last=False, persistent_workers=None):
    nw = _optimal_workers(num_workers)
    pw = (nw > 0) if persistent_workers is None else persistent_workers
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(pw and nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
        collate_fn=collate_fn,
    )


# Text / LLM Datasets 

class TextDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int):
        self.tokens  = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self): return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class ChatDataset(Dataset):
    """Standard chat pairs (no thinking)."""

    def __init__(self, pairs: List[dict], tokenizer, seq_len: int):
        self.samples = []
        for item in pairs:
            prompt   = item.get("prompt", item.get("input", ""))
            response = item.get("response", item.get("output", ""))
            user_tag = getattr(tokenizer, "USER_TAG", None)
            asst_tag = getattr(tokenizer, "ASST_TAG", None)
            if user_tag is not None:
                full = f"<|user|>{prompt}<|assistant|>{response}"
            else:
                full = f"User: {prompt}\nAssistant: {response}"
            toks = tokenizer.encode(full)
            if len(toks) <= seq_len + 1:
                self.samples.append(toks)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        t = self.samples[idx]
        return torch.tensor(t[:-1], dtype=torch.long), torch.tensor(t[1:], dtype=torch.long)


class ThinkingDataset(Dataset):
    """
    Chat pairs with chain-of-thought thinking traces.

    Expected JSONL format:
      {"prompt": "...", "thinking": "...", "response": "..."}

    If "thinking" is present the sample becomes:
      <|user|>PROMPT<|assistant|><think>THINKING</think>RESPONSE

    If "thinking" is absent, behaves like ChatDataset.
    """

    def __init__(self, pairs: List[dict], tokenizer, seq_len: int):
        self.samples = []
        self.masks   = []   # thinking masks per sample

        for item in pairs:
            prompt   = item.get("prompt",   item.get("input",    ""))
            response = item.get("response", item.get("output",   ""))
            thinking = item.get("thinking", "")

            if thinking:
                full = f"<|user|>{prompt}<|assistant|><think>{thinking}</think>{response}"
            else:
                full = f"<|user|>{prompt}<|assistant|>{response}"

            toks = tokenizer.encode(full)
            if len(toks) <= seq_len + 1:
                self.samples.append(toks)
                if hasattr(tokenizer, "build_thinking_mask"):
                    self.masks.append(tokenizer.build_thinking_mask(toks))
                else:
                    self.masks.append([0] * len(toks))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        t = self.samples[idx]; m = self.masks[idx]
        return (torch.tensor(t[:-1], dtype=torch.long),
                torch.tensor(t[1:],  dtype=torch.long),
                torch.tensor(m[:-1], dtype=torch.long))   # thinking mask aligned to input


# Vision Datasets 

class ImageFolderFlat(Dataset):
    """Images only (for VAE training)."""
    EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, root: str, img_size: int = 128):
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if os.path.splitext(f)[1].lower() in self.EXTS]
        self.img_size = img_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(self.paths[idx]).convert("RGB")
            tfm = T.Compose([T.Resize((self.img_size, self.img_size)), T.ToTensor(),
                              T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
            return tfm(img)
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size)


class VisionChatDataset(Dataset):
    """
    Image + text pairs for PicoVLM training.

    Expects a JSONL file where each line is:
      {"image": "path/to/img.jpg", "prompt": "...", "response": "...",
       "thinking": "..."  (optional)}

    Returns (image_tensor, input_tokens, target_tokens, thinking_mask).
    """

    def __init__(self, jsonl_path: str, tokenizer, seq_len: int, img_size: int = 128):
        self.records  = []
        self.seq_len  = seq_len
        self.img_size = img_size
        self.tokenizer = tokenizer

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    prompt   = item.get("prompt", "")
                    response = item.get("response", "")
                    thinking = item.get("thinking", "")
                    img_path = item.get("image", "")

                    if thinking:
                        text = f"<|user|>{prompt}<|assistant|><think>{thinking}</think>{response}"
                    else:
                        text = f"<|user|>{prompt}<|assistant|>{response}"

                    toks = tokenizer.encode(text)
                    if len(toks) <= seq_len + 1 and img_path:
                        self.records.append({"image": img_path, "tokens": toks,
                                             "thinking": thinking})

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Load image
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(rec["image"]).convert("RGB")
            tfm = T.Compose([T.Resize((self.img_size, self.img_size)), T.ToTensor(),
                              T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
            img_tensor = tfm(img)
        except Exception:
            img_tensor = torch.zeros(3, self.img_size, self.img_size)

        t = rec["tokens"]
        if hasattr(self.tokenizer, "build_thinking_mask"):
            m = self.tokenizer.build_thinking_mask(t)
        else:
            m = [0] * len(t)

        x  = torch.tensor(t[:-1], dtype=torch.long)
        y  = torch.tensor(t[1:],  dtype=torch.long)
        tm = torch.tensor(m[:-1], dtype=torch.long)
        return img_tensor, x, y, tm


#  Collation 
def collate_pad(batch, pad_id: int = 0):
    
    has_mask = len(batch[0]) == 3
    if has_mask:
        xs, ys, ms = zip(*batch)
    else:
        xs, ys = zip(*batch); ms = None

    max_len = max(x.size(0) for x in xs)
    pad = lambda seqs, val: torch.stack([
        torch.nn.functional.pad(s, (0, max_len - s.size(0)), value=val) for s in seqs])

    xp = pad(xs, pad_id); yp = pad(ys, -1)
    if ms is not None:
        mp = pad(ms, 0)
        return xp, yp, mp
    return xp, yp

def collate_vlm(batch, pad_id: int = 0):
    """Collate (img, x, y, mask) batches."""
    imgs, xs, ys, ms = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    pad = lambda seqs, val: torch.stack([
        torch.nn.functional.pad(s, (0, max_len - s.size(0)), value=val) for s in seqs])
    return torch.stack(imgs), pad(xs, pad_id), pad(ys, -1), pad(ms, 0)


# DataLoader factories 

def make_text_loader(tokens, seq_len, batch_size, shuffle=True):
    return _make_loader(TextDataset(tokens, seq_len), batch_size, shuffle,
                        drop_last=True)

def make_chat_loader(pairs, tokenizer, seq_len, batch_size, shuffle=True):
    ds = ChatDataset(pairs, tokenizer, seq_len)
    return _make_loader(ds, batch_size, shuffle,
                        collate_fn=lambda b: collate_pad(b))

def make_thinking_loader(pairs, tokenizer, seq_len, batch_size, shuffle=True):
    """Loader that also returns thinking masks for loss weighting."""
    ds = ThinkingDataset(pairs, tokenizer, seq_len)
    return _make_loader(ds, batch_size, shuffle,
                        collate_fn=lambda b: collate_pad(b))

def make_image_loader(root, img_size, batch_size, shuffle=True):
    return _make_loader(ImageFolderFlat(root, img_size), batch_size, shuffle,
                        drop_last=True)

def make_vlm_loader(jsonl_path, tokenizer, seq_len, img_size, batch_size, shuffle=True):
    """Loader for vision-language training data."""
    ds = VisionChatDataset(jsonl_path, tokenizer, seq_len, img_size)
    return _make_loader(ds, batch_size, shuffle,
                        collate_fn=lambda b: collate_vlm(b))


# File helpers 

def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: items.append(json.loads(line))
    return items

def load_txt(path: str) -> str:
    with open(path) as f: return f.read()
