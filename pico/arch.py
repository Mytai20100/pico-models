import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class PicoConfig:
    vocab_size: int = 32000
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_mult: float = 2.67
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_flash: bool = True
    img_patch_size: int = 16
    img_size: int = 128
    img_channels: int = 3
    latent_dim: int = 64
    model_type: str = "llm"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: xq_.shape[2]]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GQA(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.use_flash = cfg.use_flash

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # apply_rotary_emb expects (B, n_heads, T, head_dim) and returns same
        xq, xk = apply_rotary_emb(xq.transpose(1, 2), xk.transpose(1, 2), freqs_cis)
        # xq: (B, n_heads, T, head_dim), xk: (B, n_kv_heads, T, head_dim)
        xv = xv.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        if kv_cache is not None:
            xk = torch.cat([kv_cache[0], xk], dim=2)
            xv = torch.cat([kv_cache[1], xv], dim=2)
        new_cache = (xk, xv)

        # Expand kv heads to match q heads — dim=1 is the heads dimension
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=(mask is None),
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.drop(scores)
            out = torch.matmul(scores, xv)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class PicoBlock(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.attn = GQA(cfg)
        self.ffn = SwiGLU(cfg)
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        h, cache = self.attn(self.norm1(x), freqs_cis, mask, kv_cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, cache


class PicoLLM(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([PicoBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.freqs_cis = precompute_freqs_cis(
            cfg.dim // cfg.n_heads, cfg.max_seq_len * 2
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None, kv_caches=None):
        B, T = tokens.shape
        device = tokens.device
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[:T]

        x = self.embed(tokens)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, c = layer(x, freqs_cis, kv_cache=cache)
            new_caches.append(c)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=128, temperature=1.0, top_p=0.9, stop_ids=None):
        self.eval()
        kv_caches = None
        for _ in range(max_new_tokens):
            inp = tokens if kv_caches is None else tokens[:, -1:]
            logits, _, kv_caches = self(inp, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if stop_ids and next_tok.item() in stop_ids:
                break
        return tokens


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PicoVAE(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.img_patch_size
        n_patches = (cfg.img_size // p) ** 2

        self.patch_embed = PatchEmbed(cfg.img_size, p, cfg.img_channels, cfg.dim)

        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, cfg.dim * 2, dropout=cfg.dropout, batch_first=True, norm_first=True)
            for _ in range(cfg.n_layers // 2)
        ])
        self.to_mu = nn.Linear(cfg.dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.dim, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.dim)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, cfg.dim * 2, dropout=cfg.dropout, batch_first=True, norm_first=True)
            for _ in range(cfg.n_layers // 2)
        ])
        self.to_pixels = nn.Linear(cfg.dim, p * p * cfg.img_channels)
        self.n_patches = n_patches
        self.patch_size = p
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, cfg.dim) * 0.02)

    def encode(self, x):
        h = self.patch_embed(x) + self.pos_embed
        for blk in self.encoder_blocks:
            h = blk(h)
        h = h.mean(dim=1)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        B = z.shape[0]
        h = self.from_latent(z).unsqueeze(1).expand(-1, self.n_patches, -1)
        h = h + self.pos_embed
        for blk in self.decoder_blocks:
            h = blk(h)
        pixels = self.to_pixels(h)
        p = self.patch_size
        side = int(self.n_patches ** 0.5)
        pixels = pixels.view(B, side, side, p, p, self.cfg.img_channels)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4).contiguous()
        pixels = pixels.view(B, self.cfg.img_channels, side * p, side * p)
        return torch.sigmoid(pixels)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + 0.001 * kl
        return recon, loss, mu, logvar

    @torch.no_grad()
    def sample(self, n=1, device="cpu"):
        self.eval()
        z = torch.randn(n, self.cfg.latent_dim, device=device)
        return self.decode(z)
