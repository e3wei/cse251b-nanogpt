import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def get_cos_sin(self, seq_len: int, device, dtype):
        if (
            self._cos_cached is None
            or seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
            self._seq_len_cached = seq_len
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base)
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.ffn_mult * config.n_embd)
        if config.ffn_dim_multiple_of > 0:
            m = config.ffn_dim_multiple_of
            hidden_dim = ((hidden_dim + m - 1) // m) * m
        self.activation = config.activation
        if self.activation == "swiglu":
            self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.w2.NANOGPT_SCALE_INIT = 1
        elif self.activation == "relu2":
            self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.w2.NANOGPT_SCALE_INIT = 1
        else:
            raise ValueError(f"unsupported activation: {self.activation}")

    def forward(self, x):
        if self.activation == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = F.relu(self.w1(x))
        x = x * x
        return self.w2(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        if norm_cls is RMSNorm:
            self.ln_1 = norm_cls(config.n_embd, eps=config.norm_eps)
            self.ln_2 = norm_cls(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_1 = norm_cls(config.n_embd, eps=config.norm_eps, bias=config.bias)
            self.ln_2 = norm_cls(config.n_embd, eps=config.norm_eps, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    rope_base: int = 10000
    activation: str = "swiglu"
    ffn_mult: float = 8.0 / 3.0
    ffn_dim_multiple_of: int = 64
    qk_norm: bool = True


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.norm_eps)
                if config.norm_type == "rmsnorm"
                else nn.LayerNorm(config.n_embd, eps=config.norm_eps, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"cannot forward sequence of length {T}, block size is {self.config.block_size}"
        )
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint and "config" in checkpoint:
        config = GPTConfig(**checkpoint["config"])
        state_dict = checkpoint["model"]
    else:
        config = GPTConfig()
        state_dict = checkpoint
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
