"""Character-level CTC vocabulary (29 tokens)."""


class Vocab:
    def __init__(self):
        self.chars = " abcdefghijklmnopqrstuvwxyz'"
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}
        self.c2i["<b>"] = 0
        self.i2c = {v: k for k, v in self.c2i.items()}
        self.size = len(self.c2i)

    def encode(self, text):
        return [self.c2i.get(c, 0) for c in text.lower() if c in self.c2i]

    def decode_greedy(self, ids):
        out, prev = [], -1
        for i in ids:
            if i == 0 or i == prev:
                prev = i
                continue
            if i in self.i2c and self.i2c[i] != "<b>":
                out.append(self.i2c[i])
            prev = i
        return ''.join(out)

    def get_labels_for_decoder(self):
        labels = []
        for idx in range(self.size):
            char = self.i2c.get(idx, "")
            labels.append("" if char == "<b>" else char)
        return labels

    def __repr__(self):
        return f"Vocab(size={self.size})"
