"""Deterministic character-level tokenizer.

Vocabulary: every distinct character in the training split, sorted by code point.
Special tokens: none. Text is modelled as one contiguous stream, so BOS/EOS/PAD are
not needed. Characters outside the vocabulary are rejected, never silently dropped.
"""
from __future__ import annotations


class UnknownCharacterError(ValueError):
    def __init__(self, chars: list[str]):
        self.chars = chars
        super().__init__(f"Characters not in the training vocabulary: {chars!r}")


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("Vocabulary contains duplicates")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def unknown_chars(self, text: str) -> list[str]:
        return sorted({ch for ch in text if ch not in self.stoi})

    def encode(self, text: str) -> list[int]:
        missing = self.unknown_chars(text)
        if missing:
            raise UnknownCharacterError(missing)
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        if data.get("type") != "char":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')}")
        return cls(list(data["chars"]))
