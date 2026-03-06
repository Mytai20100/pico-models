import os
import json
import random
from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class ChatDataset(Dataset):
    def __init__(self, pairs: List[dict], tokenizer, seq_len: int):
        self.samples = []
        for item in pairs:
            prompt = item.get("prompt", item.get("input", ""))
            response = item.get("response", item.get("output", ""))
            full = f"User: {prompt}\nAssistant: {response}"
            toks = tokenizer.encode(full)
            if len(toks) <= seq_len + 1:
                self.samples.append(toks)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        toks = self.samples[idx]
        return torch.tensor(toks[:-1], dtype=torch.long), torch.tensor(toks[1:], dtype=torch.long)


def collate_pad(batch, pad_id: int = 0):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xp = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_id) for x in xs])
    yp = torch.stack([torch.nn.functional.pad(y, (0, max_len - y.size(0)), value=-1) for y in ys])
    return xp, yp


class ImageFolderFlat(Dataset):
    EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, root: str, img_size: int = 128, transform=None):
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in self.EXTS
        ]
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                tfm = T.Compose([T.Resize((self.img_size, self.img_size)), T.ToTensor()])
                img = tfm(img)
            return img
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size)


def make_text_loader(tokens, seq_len, batch_size, shuffle=True, num_workers=2):
    ds = TextDataset(tokens, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def make_chat_loader(pairs, tokenizer, seq_len, batch_size, shuffle=True):
    ds = ChatDataset(pairs, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: collate_pad(b), num_workers=2, pin_memory=True)


def make_image_loader(root, img_size, batch_size, shuffle=True):
    ds = ImageFolderFlat(root, img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=True)


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_txt(path: str) -> str:
    with open(path) as f:
        return f.read()
