"""
EigenWave-ASR Model Architecture
=================================
Robin Frontend → 12× Conformer Blocks → CTC (27.8M params)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiScaleRobinFeatures(nn.Module):
    def __init__(self, n_mels=80, d_model=384):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160,
            n_mels=n_mels, normalized=True
        )
        self.scales = [1, 3, 5]
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.ones(n_mels) * 0.6) for _ in self.scales
        ])
        self.betas = nn.ParameterList([
            nn.Parameter(torch.ones(n_mels) * 0.25) for _ in self.scales
        ])
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(n_mels) * 0.15) for _ in self.scales
        ])
        self.scale_conv = nn.Conv1d(n_mels * len(self.scales), d_model, 1)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def _safe_pad(self, x, pad_size):
        added_dims = 0
        while x.dim() < 3:
            x = x.unsqueeze(0)
            added_dims += 1
        try:
            x_padded = F.pad(x, (pad_size, pad_size), mode='replicate')
        except (NotImplementedError, RuntimeError):
            left = x[..., :1].expand(*x.shape[:-1], pad_size)
            right = x[..., -1:].expand(*x.shape[:-1], pad_size)
            x_padded = torch.cat([left, x, right], dim=-1)
        for _ in range(added_dims):
            x_padded = x_padded.squeeze(0)
        return x_padded

    def compute_robin(self, mel, scale, alpha, beta, gamma):
        mel_pad = self._safe_pad(mel, scale)
        d1 = (mel_pad[..., 2*scale:] - mel_pad[..., :-2*scale]) / (2 * scale)
        d2 = (mel_pad[..., 2*scale:] - 2*mel_pad[..., scale:-scale] +
              mel_pad[..., :-2*scale]) / (scale * scale)
        d1 = d1[..., :mel.size(-1)]
        d2 = d2[..., :mel.size(-1)]
        alpha = alpha.view(1, -1, 1)
        beta = beta.view(1, -1, 1)
        gamma = gamma.view(1, -1, 1)
        return alpha * mel + beta * d1 + gamma * d2

    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)
        elif wav.dim() == 3 and wav.shape[1] > 1:
            wav = wav.mean(dim=1, keepdim=True)
        mel = self.mel(wav).squeeze(1)
        while mel.dim() > 3:
            mel = mel.mean(dim=1)
        mel = torch.log(mel.clamp(min=1e-5))
        robin_features = []
        for i, scale in enumerate(self.scales):
            robin = self.compute_robin(
                mel, scale, self.alphas[i], self.betas[i], self.gammas[i]
            )
            robin_features.append(robin)
        multi_scale = torch.cat(robin_features, dim=1)
        fused = self.scale_conv(multi_scale)
        out = self.conv(fused)
        out = out.transpose(1, 2)
        return self.dropout(self.norm(out))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=8000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        self._init_cache()

    def _init_cache(self):
        t = torch.arange(self.max_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))

    def forward(self, x, seq_len):
        if seq_len > self.max_len:
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(x.device))
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, d_model * 2)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pointwise1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        x = self.pointwise2(x)
        return self.dropout(x)


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head)
        self.conv = ConvModule(d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = self.rope(x, T)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = x + out
        x = x + self.conv(x)
        x = x + self.ffn(self.norm2(x))
        return x


class EnhancedHybridASR(nn.Module):
    def __init__(self, d_model=384, n_layers=12, n_heads=8,
                 vocab_size=29, dropout=0.1):
        super().__init__()
        self.frontend = MultiScaleRobinFeatures(80, d_model)
        self.spec_aug = nn.Identity()
        self.encoder = nn.ModuleList([
            EnhancedTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, wav):
        x = self.frontend(wav)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        return self.out(x)

    def get_robin_stats(self):
        stats = {}
        for i, scale in enumerate(self.frontend.scales):
            stats[f'scale_{scale}'] = {
                'alpha': self.frontend.alphas[i].mean().item(),
                'beta': self.frontend.betas[i].mean().item(),
                'gamma': self.frontend.gammas[i].mean().item(),
            }
        return stats

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)