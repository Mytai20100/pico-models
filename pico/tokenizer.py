"""
pico/tokenizer.py  –  Tokenizers with Thinking + Vision special tokens (v2)

Special tokens added:
  <think>   – begin chain-of-thought reasoning block
  </think>  – end chain-of-thought reasoning block
  <img>     – begin image placeholder in text
  </img>    – end image placeholder in text
  <|user|>  – speaker tags for chat format
  <|assistant|>
  <|system|>
"""
import os, json, re
from typing import List, Optional


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>",
                  "<think>", "</think>",
                  "<img>", "</img>",
                  "<|user|>", "<|assistant|>", "<|system|>"]


class CharTokenizer:
    PAD = 0; BOS = 1; EOS = 2; UNK = 3
    THINK_START = 4; THINK_END = 5
    IMG_START   = 6; IMG_END   = 7
    USER_TAG    = 8; ASST_TAG  = 9; SYS_TAG = 10

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self._ch2id: dict = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self._id2ch: dict = {i: t for t, i in self._ch2id.items()}

    def train(self, texts: List[str]):
        freq: dict = {}
        for t in texts:
            for c in t:
                freq[c] = freq.get(c, 0) + 1
        for c, _ in sorted(freq.items(), key=lambda x: -x[1]):
            if len(self._ch2id) >= self.vocab_size:
                break
            if c not in self._ch2id:
                idx = len(self._ch2id)
                self._ch2id[c] = idx
                self._id2ch[idx] = c

    def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:
        # Handle multi-char special tokens first
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for sp in sorted(self._ch2id.keys(), key=len, reverse=True):
                if len(sp) > 1 and text[i:i+len(sp)] == sp:
                    tokens.append(self._ch2id[sp])
                    i += len(sp); matched = True; break
            if not matched:
                tokens.append(self._ch2id.get(text[i], self.UNK)); i += 1
        if bos: tokens = [self.BOS] + tokens
        if eos: tokens = tokens + [self.EOS]
        return tokens

    # All special token IDs: 0=PAD 1=BOS 2=EOS 3=UNK 4=<think> 5=</think>
    # 6=<img> 7=</img> 8=<|user|> 9=<|assistant|> 10=<|system|>
    _N_SPECIAL = len(SPECIAL_TOKENS)

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if skip_special and i < self._N_SPECIAL:
                # Skip ALL special tokens (PAD, BOS, UNK, <think>, <|user|>, etc.)
                continue
            out.append(self._id2ch.get(i, "?"))
        return "".join(out)

    def decode_with_thinking(self, ids: List[int]) -> tuple:
        """Returns (answer_text, thinking_text). Skips all special tokens except think tags."""
        # Build full string preserving <think></think> markers but skipping other specials
        parts = []
        for i in ids:
            if i == self.EOS:
                break
            if i < self._N_SPECIAL:
                # Keep think markers as literal text so regex can find them
                if i == self.THINK_START:
                    parts.append("<think>")
                elif i == self.THINK_END:
                    parts.append("</think>")
                # All other specials (user/assistant/img tags) silently skipped
                continue
            parts.append(self._id2ch.get(i, "?"))
        full = "".join(parts)
        think_pat = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        thinking = " ".join(m.group(1).strip() for m in think_pat.finditer(full))
        answer   = think_pat.sub("", full).strip()
        return answer, thinking

    def build_thinking_mask(self, ids: List[int]) -> List[int]:
        """1 inside <think>…</think>, 0 outside."""
        mask = []; inside = False
        for i in ids:
            if i == self.THINK_START: inside = True
            mask.append(1 if inside else 0)
            if i == self.THINK_END:   inside = False
        return mask

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self._ch2id, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f: data = json.load(f)
        t = cls(data["vocab_size"])
        t._ch2id = {k: int(v) for k, v in data["vocab"].items()}
        t._id2ch = {int(v): k for k, v in data["vocab"].items()}
        return t

    def __len__(self): return len(self._ch2id)


# ── SentencePiece wrapper (optional) ──

try:
    import sentencepiece as spm

    class SPTokenizer:
        PAD = 0; BOS = 1; EOS = 2; UNK = 3
        # Special IDs injected after base vocab
        THINK_START = None; THINK_END = None
        IMG_START   = None; IMG_END   = None

        def __init__(self, model_path: Optional[str] = None):
            self._sp = None
            if model_path and os.path.exists(model_path):
                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(model_path)
                # Map special tokens to IDs
                for attr, tok in [("THINK_START","<think>"),("THINK_END","</think>"),
                                   ("IMG_START","<img>"),("IMG_END","</img>")]:
                    tid = self._sp.PieceToId(tok)
                    setattr(self, attr, tid if tid != 0 else None)

        def train(self, input_file: str, model_prefix: str, vocab_size: int = 4096):
            special_str = ",".join(SPECIAL_TOKENS[4:])   # extra special tokens
            spm.SentencePieceTrainer.Train(
                f"--input={input_file} --model_prefix={model_prefix} "
                f"--vocab_size={vocab_size} --character_coverage=0.9995 "
                f"--model_type=bpe --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 "
                f"--user_defined_symbols={special_str}"
            )
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(f"{model_prefix}.model")

        def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:
            ids = self._sp.EncodeAsIds(text)
            if bos: ids = [self.BOS] + ids
            if eos: ids = ids + [self.EOS]
            return ids

        def decode(self, ids: List[int], skip_special: bool = True) -> str:
            ids = [i for i in ids if i not in (self.BOS, self.PAD)]
            if self.EOS in ids: ids = ids[:ids.index(self.EOS)]
            return self._sp.DecodeIds(ids)

        def build_thinking_mask(self, ids: List[int]) -> List[int]:
            mask = []; inside = False
            for i in ids:
                if i == self.THINK_START: inside = True
                mask.append(1 if inside else 0)
                if i == self.THINK_END:   inside = False
            return mask

        def save(self, path: str): pass

        @classmethod
        def load(cls, path: str) -> "SPTokenizer": return cls(path)

        def __len__(self): return self._sp.GetPieceSize() if self._sp else 0

except ImportError:
    SPTokenizer = None


def get_tokenizer(kind: str = "char", **kwargs):
    if kind == "sp" and SPTokenizer is not None:
        return SPTokenizer(**kwargs)
    return CharTokenizer(**kwargs)
