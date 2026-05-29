import json
import re
from pathlib import Path
import urllib.request


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class BPETokenizer:
    def __init__(self, vocab, merges):
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {tuple(p.split()): i for i, p in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")

    def bpe(self, token):
        word = list(token)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if pair not in self.bpe_ranks:
                break
            i = 0
            new = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new.append("".join(pair))
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        return " ".join(word)

    def encode(self, text):
        text = text.encode("utf-8")
        text = "".join(self.byte_encoder[b] for b in text)
        words = re.findall(self.pat, text)
        tokens = []
        for word in words:
            token_str = self.bpe(word)
            for t in token_str.split():
                tokens.append(self.encoder[t])
        return tokens

    def decode(self, ids):
        text = "".join(self.decoder[id] for id in ids)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        return text

    @classmethod
    def from_pretrained(cls, name="gpt2"):
        cache = Path.home() / ".cache" / "gpt-from-scratch"
        cache.mkdir(parents=True, exist_ok=True)

        path = cache / f"{name.replace('/', '-')}_tokenizer.json"

        if not path.exists():
            hf_name = name if "/" in name else f"openai-community/{name}"
            url = f"https://huggingface.co/{hf_name}/resolve/main/tokenizer.json"
            print(f"Downloading tokenizer.json...")
            urllib.request.urlretrieve(url, path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(data["model"]["vocab"], data["model"]["merges"])
