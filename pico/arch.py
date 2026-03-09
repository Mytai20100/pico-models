from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


# Config

@dataclass
class PicoConfig:
    # LLM core
    vocab_size:     int   = 32000
    dim:            int   = 256
    n_layers:       int   = 6
    n_heads:        int   = 8
    n_kv_heads:     int   = 2
    ffn_mult:       float = 2.67
    max_seq_len:    int   = 512
    dropout:        float = 0.0
    tie_embeddings: bool  = True
    use_flash:      bool  = True
    # Vision
    img_patch_size: int   = 16
    img_size:       int   = 128
    img_channels:   int   = 3
    # VAE
    latent_dim:     int   = 64
    # Thinking
    use_thinking:   bool  = False
    think_start_id: int   = -1
    think_end_id:   int   = -1
    # Model type
    model_type:     str   = "llm"   # "llm" | "vlm" | "vae"


# RoPE 

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    T = xq.shape[2]
    fc = freqs_cis[:T].unsqueeze(0).unsqueeze(0)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    return (torch.view_as_real(xq_ * fc).flatten(3).type_as(xq),
            torch.view_as_real(xk_ * fc).flatten(3).type_as(xk))


# Primitives 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        hidden = (int(cfg.dim * cfg.ffn_mult) + 7) // 8 * 8  # 8-aligned for tensor cores
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class GQA(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.n_heads = cfg.n_heads; self.n_kv = cfg.n_kv_heads
        self.hd = cfg.dim // cfg.n_heads; self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.dim, bias=False)
        self.drop_p = cfg.dropout; self.use_flash = cfg.use_flash

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B,T,self.n_heads,self.hd).transpose(1,2)
        k = self.wk(x).view(B,T,self.n_kv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,T,self.n_kv,self.hd).transpose(1,2)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                      dropout_p=self.drop_p if self.training else 0.0, is_causal=(mask is None))
        else:
            sc = (torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.hd))
            if mask is not None: sc = sc + mask
            sc = F.softmax(sc.float(), dim=-1).type_as(q)
            if self.drop_p > 0 and self.training: sc = F.dropout(sc, p=self.drop_p)
            out = torch.matmul(sc, v)
        return self.wo(out.transpose(1,2).contiguous().view(B,T,-1)), new_cache

class PicoBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GQA(cfg); self.ffn = SwiGLU(cfg)
        self.norm1 = RMSNorm(cfg.dim); self.norm2 = RMSNorm(cfg.dim)
    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        h, c = self.attn(self.norm1(x), freqs_cis, mask, kv_cache)
        x = x + h; x = x + self.ffn(self.norm2(x))
        return x, c

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


# PicoLLM 

class PicoLLM(nn.Module):
   
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.cfg = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList([PicoBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.vocab_size if not cfg.tie_embeddings else cfg.vocab_size,
                                 cfg.vocab_size, bias=False)
        # proper tied linear
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.register_buffer("freqs_cis",
            precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len * 2), persistent=False)
        self.apply(_init_weights)

    @property
    def num_params(self): return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None, kv_caches=None, thinking_mask=None):
        B, T = tokens.shape
        x = self.embed(tokens)
        fc = self.freqs_cis[:T]
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, c = layer(x, fc, kv_cache=kv_caches[i] if kv_caches else None)
            new_caches.append(c)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            fl = logits.view(-1, logits.size(-1)); ft = targets.view(-1)
            if thinking_mask is not None and self.cfg.use_thinking:
                fm = thinking_mask.view(-1).float()
                tl = F.cross_entropy(fl, ft, ignore_index=-1, reduction="none")
                valid = (ft != -1).float()
                loss = (tl * (1.0 + fm) * valid).sum() / (valid.sum() + 1e-9)
            else:
                loss = F.cross_entropy(fl, ft, ignore_index=-1)
        return logits, loss, new_caches

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens=128, temperature=1.0, top_p=0.9,
                 stop_ids=None, show_thinking=False):
        self.eval(); kv_caches = None
        for _ in range(max_new_tokens):
            inp = tokens if kv_caches is None else tokens[:, -1:]
            logits, _, kv_caches = self(inp, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = float("-inf")
                logits.scatter_(1, si, sl)
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if stop_ids and next_tok.item() in stop_ids: break
        return tokens


# Vision Encoder

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.n_patches = (img_size // patch_size) ** 2
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class VisionEncoder(nn.Module):

    def __init__(self, cfg: PicoConfig):
        super().__init__()
        n_patches   = (cfg.img_size // cfg.img_patch_size) ** 2
        n_vis_layers = max(2, cfg.n_layers // 3)
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.img_patch_size, cfg.img_channels, cfg.dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches + 1, cfg.dim) * 0.02)
        enc = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, int(cfg.dim * cfg.ffn_mult),
                                         dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.blocks  = nn.TransformerEncoder(enc, num_layers=n_vis_layers)
        self.norm    = RMSNorm(cfg.dim)
        self.proj    = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.n_patches = n_patches

    def forward(self, imgs):
        """imgs: (B,C,H,W) → (B, N_patches+1, D)"""
        B = imgs.size(0)
        x = self.patch_embed(imgs)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1) + self.pos_embed
        return self.proj(self.norm(self.blocks(x)))


# PicoVLM
class PicoVLM(nn.Module):

    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.cfg            = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.embed          = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers         = nn.ModuleList([PicoBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm           = RMSNorm(cfg.dim)
        self.lm_head        = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.register_buffer("freqs_cis",
            precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len * 2 + 300), persistent=False)
        self.apply(_init_weights)

    @property
    def num_params(self): return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None, images=None, thinking_mask=None):
        text_emb = self.embed(tokens)
        if images is not None:
            vis = self.vision_encoder(images)         # (B, N_vis, D)
            N_vis = vis.size(1)
            x = torch.cat([vis, text_emb], dim=1)
        else:
            N_vis = 0; x = text_emb

        fc = self.freqs_cis[:x.size(1)]
        new_caches = []
        for layer in self.layers:
            x, c = layer(x, fc)
            new_caches.append(c)

        text_out = self.norm(x[:, N_vis:, :])
        logits   = self.lm_head(text_out)

        loss = None
        if targets is not None:
            fl = logits.view(-1, logits.size(-1)); ft = targets.view(-1)
            if thinking_mask is not None and self.cfg.use_thinking:
                fm    = thinking_mask.view(-1).float()
                tl    = F.cross_entropy(fl, ft, ignore_index=-1, reduction="none")
                valid = (ft != -1).float()
                loss  = (tl * (1.0 + fm) * valid).sum() / (valid.sum() + 1e-9)
            else:
                loss = F.cross_entropy(fl, ft, ignore_index=-1)
        return logits, loss, new_caches

    @torch.inference_mode()
    def generate(self, tokens, images=None, max_new_tokens=128,
                 temperature=1.0, top_p=0.9, stop_ids=None):
        self.eval()
        vis_tokens = self.vision_encoder(images) if images is not None else None
        kv_caches  = None

        for step in range(max_new_tokens):
            if kv_caches is None:
                emb = self.embed(tokens)
                if vis_tokens is not None:
                    emb = torch.cat([vis_tokens, emb], dim=1)
            else:
                emb = self.embed(tokens[:, -1:])

            fc = self.freqs_cis[:emb.size(1)]
            x  = emb; new_caches = []
            for i, layer in enumerate(self.layers):
                x, c = layer(x, fc, kv_cache=kv_caches[i] if kv_caches else None)
                new_caches.append(c)
            kv_caches = new_caches

            N_vis  = (vis_tokens.size(1) if (vis_tokens is not None and step == 0) else 0)
            logits = self.lm_head(self.norm(x[:, N_vis:, :]))[:, -1, :] / max(temperature, 1e-5)

            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = float("-inf")
                logits.scatter_(1, si, sl)

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens   = torch.cat([tokens, next_tok], dim=1)
            if stop_ids and next_tok.item() in stop_ids: break
        return tokens


# PicoVAE

class PicoVAE(nn.Module):
    def __init__(self, cfg: PicoConfig):
        super().__init__()
        self.cfg = cfg; p = cfg.img_patch_size
        n_patches = (cfg.img_size // p) ** 2
        n_enc = max(2, cfg.n_layers // 2); n_dec = max(2, cfg.n_layers // 2)
        self.patch_embed  = PatchEmbed(cfg.img_size, p, cfg.img_channels, cfg.dim)
        self.pos_embed    = nn.Parameter(torch.randn(1, n_patches, cfg.dim) * 0.02)
        enc = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, cfg.dim*2,
                                         dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(enc, num_layers=n_enc)
        self.to_mu = nn.Linear(cfg.dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.dim, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.dim)
        dec = nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, cfg.dim*2,
                                         dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.decoder_blocks = nn.TransformerEncoder(dec, num_layers=n_dec)
        self.to_pixels = nn.Linear(cfg.dim, p * p * cfg.img_channels)
        self.n_patches = n_patches; self.patch_size = p
        self.apply(_init_weights)

    @property
    def num_params(self): return sum(p.numel() for p in self.parameters())

    def encode(self, x):
        h = self.encoder_blocks(self.patch_embed(x) + self.pos_embed).mean(dim=1)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + (0.5*logvar).exp() * torch.randn_like(mu) if self.training else mu

    def decode(self, z):
        B = z.size(0); p = self.patch_size; side = int(self.n_patches ** 0.5)
        h  = self.decoder_blocks(self.from_latent(z).unsqueeze(1).expand(-1,self.n_patches,-1) + self.pos_embed)
        px = self.to_pixels(h).view(B, side, side, p, p, self.cfg.img_channels)
        return torch.sigmoid(px.permute(0,5,1,3,2,4).contiguous().view(B, self.cfg.img_channels, side*p, side*p))

    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); recon = self.decode(z)
        kl   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        loss = F.mse_loss(recon, x) + 0.001 * kl
        return recon, loss, mu, logvar

    @torch.inference_mode()
    def sample(self, n=1, device="cpu"):
        self.eval(); return self.decode(torch.randn(n, self.cfg.latent_dim, device=device))
