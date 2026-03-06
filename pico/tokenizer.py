import os
import json
import re
from typing import List, Optional


class CharTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self._ch2id: dict = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self._id2ch: dict = {v: k for k, v in self._ch2id.items()}

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
        ids = [self._ch2id.get(c, self.UNK) for c in text]
        if bos:
            ids = [self.BOS] + ids
        if eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in (self.BOS, self.PAD):
                continue
            if i == self.EOS:
                break
            out.append(self._id2ch.get(i, "?"))
        return "".join(out)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self._ch2id, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        t = cls(data["vocab_size"])
        t._ch2id = {k: int(v) for k, v in data["vocab"].items()}
        t._id2ch = {int(v): k for k, v in data["vocab"].items()}
        return t

    def __len__(self):
        return len(self._ch2id)


try:
    import sentencepiece as spm

    class SPTokenizer:
        PAD = 0
        BOS = 1
        EOS = 2
        UNK = 3

        def __init__(self, model_path: Optional[str] = None):
            self._sp = None
            if model_path and os.path.exists(model_path):
                self._sp = spm.SentencePieceProcessor()
                self._sp.Load(model_path)

        def train(self, input_file: str, model_prefix: str, vocab_size: int = 4096):
            spm.SentencePieceTrainer.Train(
                f"--input={input_file} --model_prefix={model_prefix} "
                f"--vocab_size={vocab_size} --character_coverage=0.9995 "
                f"--model_type=bpe --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3"
            )
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(f"{model_prefix}.model")

        def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:
            ids = self._sp.EncodeAsIds(text)
            if bos:
                ids = [self.BOS] + ids
            if eos:
                ids = ids + [self.EOS]
            return ids

        def decode(self, ids: List[int]) -> str:
            ids = [i for i in ids if i not in (self.BOS, self.PAD)]
            if self.EOS in ids:
                ids = ids[: ids.index(self.EOS)]
            return self._sp.DecodeIds(ids)

        def save(self, path: str):
            pass

        @classmethod
        def load(cls, path: str) -> "SPTokenizer":
            return cls(path)

        def __len__(self):
            return self._sp.GetPieceSize()

except ImportError:
    SPTokenizer = None


def get_tokenizer(kind: str = "char", **kwargs):
    if kind == "sp" and SPTokenizer is not None:
        return SPTokenizer(**kwargs)
    return CharTokenizer(**kwargs)
