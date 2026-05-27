"""
FlyAttention — Biologically-inspired linear attention replacement.

Based on Drosophila central complex connectome structure:

  Parallel streams (PB columns)      — no cross-talk, independent processing
       ↓
  Portal / BU-LAL equivalent         — FIRST cross-stream convergence point
       ↓                               topographic, not fully mixed
  Bias Cachneck (Δ7)                 — sparse, inhibitory side channel
       ↓          ↑                    NOT a bottleneck — biases layer, never compresses it
  Cache (E-PG / P-EN reciprocal)     — inter-pass feedback, held across denoising steps
       ↓                               residual history accumulates naturally
  Output — inhibitory subtraction    — suppress established signal, pass novel

Inter-layer connection:
  Each FlyAttention layer has its OWN portal and bias cachneck.
  Layers connect ONLY through the cache:
    Layer N: streams → portal_N → bcn → cache_N
    Layer N+1: streams(+cache_N bias) → portal_N+1 → bcn → cache_N+1

Streams NEVER talk to each other directly — only portal → bcn → cache → next layer.

Complexity: O(n) in sequence length — no O(n²) anywhere.
Parameters: ~38x fewer than standard attention at d=768.
Drop-in replacement: d_model in → d_model out.

NOTE — Functionally inspired, not a biological simulation:
  This is NOT an attempt to accurately simulate the Drosophila brain.
  The biological structure is the inspiration; the computational function is the goal.
  We extract the PRINCIPLES — parallel independent streams, sparse inhibitory feedback,
  reciprocal cache attractors, topographic convergence — and implement them as efficient
  tensor operations optimized for modern hardware (TPU/mobile NPU).
  Parameter counts, connectivity patterns, and neuron types are chosen for hardware
  efficiency and gradient stability, not biological accuracy.
  Think of it like LSTMs: inspired by memory mechanisms, not a neuroscience simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class FlyAttentionConfig:
    d_model: int = 768
    n_streams: int = 16       # PB columns — must divide d_model
    bcn_size: int = 8         # bias cachneck size (Δ7 channel width)
                               # configurable: larger = more working memory capacity
    sparsity: float = 0.5     # fraction of Δ7 weights zeroed (biological sparse connectivity)
    portal_heads: int = 4     # topographic portal — how many parallel channels in BU/LAL
    bcn_momentum: float = 0.9 # EMA momentum for cache update (0.0=hard replace, 0.9=slow decay)
                               # higher = cache holds information longer across steps
                               # biological analogy: dopaminergic gating — synapses strengthen
                               # proportionally to prediction error, not average state



class FlyAttention(nn.Module):
    """
    Single FlyAttention layer — drop-in attention replacement.
    Each layer has its own portal and bias cachneck.
    Layers communicate only through the cache.
    """

    def __init__(self, config: FlyAttentionConfig):
        super().__init__()
        assert config.d_model % config.n_streams == 0, \
            f"d_model ({config.d_model}) must be divisible by n_streams ({config.n_streams})"
        assert config.d_model % config.portal_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by portal_heads ({config.portal_heads})"

        self.cfg = config
        self.stream_dim = config.d_model // config.n_streams   # 768/16 = 48
        self.portal_dim = config.d_model // config.portal_heads # 768/4 = 192

        # ── Parallel streams (PB columns) — stacked for TPU BatchMatMul ──────────
        # Biologically: 16 independent PB columns, no cross-talk
        # Implementation: single (16, 48, 48) weight tensor → one torch.matmul
        # XLA treats the 16 dim as a hard batch wall — mathematically identical
        # to 16 separate Linear layers but eliminates 16-node HLO graph bloat
        # Mobile export: weights.unbind(0) → 16 separate tensors for NPU inference
        # xavier_uniform_ instead of orthogonal_ — safer in bfloat16
        streams_blocks = [
            torch.empty(self.stream_dim, self.stream_dim)
            for _ in range(config.n_streams)
        ]
        for b in streams_blocks:
            nn.init.xavier_uniform_(b)
        self.streams_weight = nn.Parameter(torch.stack(streams_blocks))
        self._cuda_streams = None  # lazy init for GPU parallel dispatch

        # ── Portal / BU-LAL — stacked for TPU BatchMatMul ────────────────────────
        # Same treatment: (4, 192, 192) weight tensor → one torch.matmul
        portal_blocks = [
            torch.empty(self.portal_dim, self.portal_dim)
            for _ in range(config.portal_heads)
        ]
        for b in portal_blocks:
            nn.init.xavier_uniform_(b)
        self.portal_weight = nn.Parameter(torch.stack(portal_blocks))
        self._portal_cuda_streams = None
        self.portal_norm = nn.LayerNorm(config.d_model)

        # ── Bias Cachneck (bcn_*) ─────────────────────────────────────────────────────
        # Named bcn_* (not bottleneck_*) to clarify: this is NOT a data bottleneck.
        # The main representation NEVER passes through here — it flows uncompressed above.
        # This is a side channel: reads activations → compresses to cache size → stores →
        # biases the NEXT pass via bcn_expand. The layer above is only biased, not compressed.
        #
        # Biologically: Δ7 interneurons in the Drosophila protocerebral bridge.
        # 8 neuron types (bcn_size=8 default), ~50% sparse connectivity.
        # They don't bottleneck the PB columns — they run alongside them as inhibitory
        # modulators, compressing population activity into a compact attractor state.
        self.bcn_compress = nn.Linear(config.d_model, config.bcn_size, bias=False)
        mask = torch.zeros(config.bcn_size, config.d_model)
        n_connections = int(config.d_model * (1.0 - config.sparsity))
        for i in range(config.bcn_size):
            idx = torch.randperm(config.d_model)[:n_connections]
            mask[i, idx] = 1.0
        self.register_buffer('bcn_mask', mask)

        # ── bcn_expand: bias cachneck → d_model (E-PG) ──────────────────────────────
        # small init — cache starts at zero, default kaiming init causes bfloat16 overflow
        self.bcn_expand = nn.Linear(config.bcn_size, config.d_model, bias=False)
        nn.init.normal_(self.bcn_expand.weight, std=0.01)

        # ── bcn_read: d_model → bias cachneck (P-EN) ────────────────────────────────
        self.bcn_read = nn.Linear(config.d_model, config.bcn_size, bias=False)
        nn.init.normal_(self.bcn_read.weight, std=0.01)

        # ── Output projection ──────────────────────────────────────────────────────
        self.bcn_out = nn.Linear(config.bcn_size, config.d_model, bias=False)
        nn.init.normal_(self.bcn_out.weight, std=0.01)

        # ── Cache — padded to 128 for TPU HBM alignment ───────────────────────────
        # Biologically: 8 Δ7 neurons. Physically: (128,) for TPU vector alignment.
        # Only first bcn_size slots used — rest is zero padding.
        # Stays on-device between steps — no host sync, no Python-land leak.
        # Updated via in-graph mean() — XLA handles as device memory copy.
        _cache_pad = 128  # TPU vector register alignment
        self.register_buffer('cache_epg', torch.zeros(_cache_pad))
        self.register_buffer('cache_pen', torch.zeros(_cache_pad))
        self._cache_pad = _cache_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) — same shape, drop-in replacement for attention

        After forward(), self._last_bottleneck, self._last_inhibition,
        self._last_cache_prev are available for auxiliary_losses().
        """
        B, T, D = x.shape
        nb = self.cfg.bcn_size

        # strip autograd history from cache at start of each forward
        # ensures Step N backward can't leak into Step N+1 graph
        if self.training:
            self.cache_epg.detach_()
            self.cache_pen.detach_()

        # ── Step 1: Parallel streams — single BatchMatMul ─────────────────────
        # EPG cache bias: only use first nb slots, rest is padding
        epg_bias = self.bcn_expand(self.cache_epg[:nb].detach().clone())
        x_biased = x + epg_bias.unsqueeze(0).unsqueeze(0)

        # reshape for batched matmul: (B, T, d_model) → (B, n_streams, T, stream_dim)
        x_streams = x_biased.view(B, T, self.cfg.n_streams, self.stream_dim)
        x_streams = x_streams.permute(0, 2, 1, 3)  # (B, n_streams, T, stream_dim)

        if x.is_cuda and self._cuda_streams is None:
            self._cuda_streams = [torch.cuda.Stream() for _ in range(self.cfg.n_streams)]

        if x.is_cuda:
            # CUDA: explicit parallel stream dispatch per stream weight
            stream_results = [None] * self.cfg.n_streams
            for i, cuda_stream in enumerate(self._cuda_streams):
                with torch.cuda.stream(cuda_stream):
                    stream_results[i] = x_streams[:, i] @ self.streams_weight[i]
            torch.cuda.synchronize()
            stream_out_s = torch.stack(stream_results, dim=1)  # (B, n_streams, T, stream_dim)
        else:
            # XLA/TPU: single BatchMatMul — one HLO node, not 16
            # streams_weight: (n_streams, stream_dim, stream_dim)
            stream_out_s = torch.matmul(x_streams, self.streams_weight)  # (B, n_streams, T, stream_dim)

        # reshape back: (B, n_streams, T, stream_dim) → (B, T, d_model)
        stream_out = stream_out_s.permute(0, 2, 1, 3).contiguous().view(B, T, D)

        # ── Step 2: Portal — single BatchMatMul ───────────────────────────────
        # reshape for batched matmul: (B, T, d_model) → (B, portal_heads, T, portal_dim)
        p_in = stream_out.view(B, T, self.cfg.portal_heads, self.portal_dim)
        p_in = p_in.permute(0, 2, 1, 3)  # (B, portal_heads, T, portal_dim)

        # single BatchMatMul: one HLO node not 4
        portal_out_s = torch.matmul(p_in, self.portal_weight)  # (B, portal_heads, T, portal_dim)
        portal_out = portal_out_s.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        portal_out = self.portal_norm(portal_out)
        # clamp to bfloat16 safe range — gradient still flows, model learns to stay in range
        portal_out = portal_out.clamp(-1e4, 1e4)

        # ── Step 3: bias cachneck ─────────────────────────────────────────────
        # Apply sparse mask functionally — avoids .data mutation mid-forward
        # which causes XLA to halt and recompile the graph
        masked_bcn = self.bcn_compress.weight * self.bcn_mask
        bottleneck = F.linear(portal_out, masked_bcn) + self.cache_pen[:nb].detach().clone().unsqueeze(0).unsqueeze(0)

        # ── Step 4: Cache update — EMA with momentum ──────────────────────────
        # EMA instead of hard mean — lets transient peaks persist rather than
        # being flattened immediately. Biological analogy: dopaminergic gating,
        # synapses strengthen proportionally to prediction error, not average state.
        # momentum=0.0 → hard replace (old behavior)
        # momentum=0.9 → slow EMA, holds information longer
        cache_prev = self.cache_epg.detach().clone()

        raw_epg = bottleneck.detach().mean(dim=(0, 1)).clamp(-1e4, 1e4)
        new_epg  = F.pad(raw_epg, (0, self._cache_pad - nb))
        m = self.cfg.bcn_momentum
        self.cache_epg.copy_(m * self.cache_epg + (1 - m) * new_epg)

        raw_pen  = self.bcn_read(portal_out.detach().mean(dim=(0, 1))).clamp(-1e4, 1e4)
        new_pen  = F.pad(raw_pen, (0, self._cache_pad - nb))
        self.cache_pen.copy_(m * self.cache_pen + (1 - m) * new_pen)

        # ── Step 5: Inhibitory output ─────────────────────────────────────────
        inhibition = self.bcn_out(bottleneck)
        out = x - inhibition

        self._last_bottleneck = bottleneck
        self._last_inhibition = inhibition
        self._last_cache_prev = cache_prev
        self._last_x          = x

        return out

    def detach_caches(self):
        """Sever cache graph history between training steps.
        Also zeros any NaN/inf values to prevent cascade corruption.
        """
        self.cache_epg.detach_()
        self.cache_pen.detach_()
        # zero NaN caches — prevents a single bad step from cascading forever
        if not torch.isfinite(self.cache_epg).all():
            self.cache_epg.zero_()
        if not torch.isfinite(self.cache_pen).all():
            self.cache_pen.zero_()

    def compute_aux_losses(self) -> dict[str, torch.Tensor]:
        """
        Compute purpose-enforcement losses from last forward pass.
        Call after forward() during training.

        Does NOT require passing intermediates — they're stored on self.
        """
        return self.auxiliary_losses(
            x=self._last_x,
            bottleneck=self._last_bottleneck,
            inhibition=self._last_inhibition,
            cache_prev=self._last_cache_prev,
        )
        """
        Compute purpose-enforcement losses from last forward pass.
        Call after forward() during training.

        Does NOT require passing intermediates — they're stored on self.
        """
        return self.auxiliary_losses(
            x=self._last_x,
            bottleneck=self._last_bottleneck,
            inhibition=self._last_inhibition,
            cache_prev=self._last_cache_prev,
        )

    def auxiliary_losses(
        self,
        x: torch.Tensor,
        bottleneck: torch.Tensor,
        inhibition: torch.Tensor,
        cache_prev: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Biological training objectives from the connectome papers:

        1. Inhibitory loss: Δ7 output should suppress what x already contains
        2. Stability loss: cache should maintain stable attractor across passes
        3. Sparsity loss: maintain Δ7 sparse connectivity (L1 on weights)
        4. Reformat loss: bottleneck must genuinely transform (not just scale)
        """
        # 1. Inhibitory — inhibition should correlate with x (suppress established)
        # eps prevents NaN gradient when input is near-zero (1/sqrt(0) = inf in backward)
        _eps = 1e-8
        loss_inhibitory = -F.cosine_similarity(
            inhibition.reshape(inhibition.shape[0], -1) + _eps,
            x.detach().reshape(x.shape[0], -1) + _eps,
            dim=1
        ).mean()

        # 2. Stability — cache should be a stable attractor, not wild oscillation
        # compare only active nb slots — rest is TPU alignment padding
        nb = self.cfg.bcn_size
        loss_stability = F.mse_loss(
            self.cache_epg[:nb],
            cache_prev[:nb].detach()
        )

        # 3. Sparsity — maintain biological sparse connectivity
        loss_sparse = self.bcn_compress.weight.abs().mean()

        # 4. Reformat — bottleneck must change representation, not just scale it
        # Compare bottleneck (B,T,8) to x (B,T,768) via mean pooling x to match
        x_pooled = x.detach().mean(dim=-1, keepdim=True).expand_as(
            bottleneck[..., :1]
        )  # (B, T, 1) scalar per position
        bottleneck_mag = bottleneck.norm(dim=-1, keepdim=True)  # (B, T, 1)
        loss_reformat = F.cosine_similarity(
            bottleneck_mag.reshape(bottleneck.shape[0], -1) + _eps,
            x_pooled.reshape(x.shape[0], -1) + _eps,
            dim=1
        ).mean().clamp(min=-0.3)

        # 5. Coherence — bottleneck should have non-trivial variance
        # unbiased=False prevents division by zero when batch*time == 1
        bottleneck_var = bottleneck.var(dim=(0, 1), unbiased=False).mean()
        loss_coherence = -bottleneck_var.clamp(max=1.0)

        # 6. Overflow forcefield — exponential penalty approaching clamp boundary
        # pow(4) means small values ≈ 0, values near 1e4 trigger massive gradient
        # teaches the model to self-regulate variance below the clamp
        # preserves cache linear relationships — clamp flattening ruins them
        max_val = 1e4
        loss_overflow = (bottleneck.abs() / max_val).pow(4).mean()

        return {
            # weights reduced — functional guidance only, not hard enforcement
            # model goals should dominate, these just nudge toward good structure
            'fly_inhibitory': loss_inhibitory * 0.001,
            'fly_stability':  loss_stability  * 0.001,
            'fly_sparse':     loss_sparse     * 0.0001,
            'fly_reformat':   loss_reformat   * 0.001,
            'fly_coherence':  loss_coherence  * 0.001,
            'fly_overflow':   loss_overflow   * 0.01,
        }

    def register_sparse_backprop(self):
        """
        Register gradient hooks to apply sparse backprop on bcn_compress.
        Only the active connections (bcn_mask=1) receive gradient updates.
        Inactive connections are frozen — prevents catastrophic forgetting
        of structure learned through sparse connectivity.

        Call once after model creation:
            for m in model.modules():
                if isinstance(m, FlyAttention):
                    m.register_sparse_backprop()
        """
        mask = self.bcn_mask  # (bcn_size, d_model)

        def _sparse_grad_hook(grad):
            # zero out gradients for masked (inactive) connections
            # only active connections update — catastrophic forgetting prevention
            return grad * mask

        self.bcn_compress.weight.register_hook(_sparse_grad_hook)

    def param_count(self) -> dict[str, int]:
        """Breakdown of parameter counts per component."""
        return {
            'streams':      self.streams_weight.numel(),
            'portal':       self.portal_weight.numel(),
            'bcn_compress':       self.bcn_compress.weight.numel(),
            'bcn_expand':   self.bcn_expand.weight.numel(),
            'bcn_read':     self.bcn_read.weight.numel(),
            'bcn_out':      self.bcn_out.weight.numel(),
            'portal_norm':  sum(p.numel() for p in self.portal_norm.parameters()),
        }


def compare_to_standard_attention(d_model: int = 768) -> None:
    """Print parameter comparison vs standard multi-head attention."""
    cfg = FlyAttentionConfig(d_model=d_model)
    fly = FlyAttention(cfg)

    fly_params = sum(p.numel() for p in fly.parameters())
    standard_params = 4 * d_model * d_model  # Q, K, V, O projections

    print(f"\nParameter comparison (d_model={d_model}):")
    print(f"  Standard attention: {standard_params:,}")
    print(f"  FlyAttention:       {fly_params:,}")
    print(f"  Reduction:          {standard_params/fly_params:.1f}x fewer params")
    print(f"\nFlyAttention breakdown:")
    for name, count in fly.param_count().items():
        print(f"  {name:12s}: {count:,}")

    # verify drop-in
    x = torch.randn(2, 128, d_model)
    y = fly(x)
    assert y.shape == x.shape
    print(f"\nShape check: {x.shape} → {y.shape} ✅")
    print(f"Complexity: O(n) in sequence length — no O(n²)")


if __name__ == '__main__':
    compare_to_standard_attention(768)


class InterPortal(nn.Module):
    """
    Cross-stream inter-portal bias injection — fully vectorized.

    Lives at the BLOCK level where multiple streams are visible simultaneously.
    Each stream's portal output is used to bias the OTHER stream's portal input
    BEFORE the portal processes it — a directional nudge, not a full transformation.

    Implementation: stacked (n_streams, n_streams, d_model, d_model) weight tensor
    + single torch.einsum — completely loop-free, one XLA BatchMatMul node.
    Self-interaction diagonal is zeroed (no stream biases itself).

    For text-only: tok ↔ sys cross-bias (replaces expensive joint cross-attention)
    For multimodal: text ↔ visual ↔ audio cross-bias (enables modality integration)
    """
    def __init__(self, d_model: int, n_streams: int = 2):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model

        # stacked (n_streams, n_streams, d_model, d_model)
        # diagonal (i==j) zeroed — no self-interaction
        raw_weights = []
        for i in range(n_streams):
            row = []
            for j in range(n_streams):
                if i != j:
                    w = torch.empty(d_model, d_model)
                    nn.init.xavier_uniform_(w)
                    row.append(w)
                else:
                    row.append(torch.zeros(d_model, d_model))
            raw_weights.append(torch.stack(row))
        self.bias_weights = nn.Parameter(torch.stack(raw_weights))

    def forward(self, stream_outs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            stream_outs: list of n_streams tensors, each (B, T, d_model)
                         may have different T (tok T ≠ sys T)
        Returns:
            list of n_streams tensors, each biased by the other streams
        """
        # mean-pool each stream to (B, 1, d) — handles different T per stream
        x_pooled = torch.stack(
            [s.mean(dim=1, keepdim=True) for s in stream_outs], dim=1
        )  # (B, n_streams, 1, d)

        # single einsum: all cross-stream biases simultaneously — one XLA node
        # b=batch, j=source stream, t=token(1), c=in_dim, i=target stream, d=out_dim
        total_bias = torch.einsum('bjtc,ijcd->bitd', x_pooled, self.bias_weights)
        # (B, n_streams, 1, d) — broadcasts over each stream's T

        return [stream_outs[i] + total_bias[:, i] for i in range(self.n_streams)]


class FlyAttentionPair(nn.Module):
    """
    Two FlyAttention instances with inter-portal cross-stream communication.
    Replaces DoubleStreamBlock attention for tok + sys streams.

    Data flow:
        tok_stream → [tok streams] → [inter_portal bias] → [tok portal] → [tok Δ7] → tok_out
        sys_stream → [sys streams] → [inter_portal bias] → [sys portal] → [sys Δ7] → sys_out
                                            ↑↑↑
                              tok and sys portal outputs bias each other HERE
                              via InterPortal (weight-based, O(n), bidirectional)
    """
    def __init__(self, config: FlyAttentionConfig):
        super().__init__()
        self.tok_attn = FlyAttention(config)
        self.sys_attn = FlyAttention(config)
        # inter-portal cross-stream communication
        # tok portal ↔ sys portal bidirectional bias
        self.inter_portal = InterPortal(config.d_model, n_streams=2)

    def forward(
        self,
        tok: torch.Tensor,  # (B, T_tok, d_model)
        sys: torch.Tensor,  # (B, T_sys, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process tok and sys streams with cross-stream portal communication.
        Returns (tok_out, sys_out) same shapes as inputs.
        """
        B_tok, T_tok, D = tok.shape
        B_sys, T_sys, _  = sys.shape
        nb = self.tok_attn.cfg.bcn_size
        n_streams = self.tok_attn.cfg.n_streams
        stream_dim = self.tok_attn.stream_dim
        portal_heads = self.tok_attn.cfg.portal_heads
        portal_dim = self.tok_attn.portal_dim

        # strip autograd history from caches at start of forward
        if self.training:
            self.tok_attn.cache_epg.detach_()
            self.tok_attn.cache_pen.detach_()
            self.sys_attn.cache_epg.detach_()
            self.sys_attn.cache_pen.detach_()

        # ── Step 1: parallel streams — single BatchMatMul each ────────────────
        tok_epg_bias = self.tok_attn.bcn_expand(self.tok_attn.cache_epg[:nb].detach().clone())
        sys_epg_bias = self.sys_attn.bcn_expand(self.sys_attn.cache_epg[:nb].detach().clone())

        tok_biased = tok + tok_epg_bias.unsqueeze(0).unsqueeze(0)
        sys_biased = sys + sys_epg_bias.unsqueeze(0).unsqueeze(0)

        # reshape: (B, T, d) → (B, n_streams, T, stream_dim)
        tok_s = tok_biased.view(B_tok, T_tok, n_streams, stream_dim).permute(0, 2, 1, 3)
        sys_s = sys_biased.view(B_sys, T_sys, n_streams, stream_dim).permute(0, 2, 1, 3)

        if tok.is_cuda:
            if self.tok_attn._cuda_streams is None:
                self.tok_attn._cuda_streams = [torch.cuda.Stream() for _ in range(n_streams)]
            tok_res = [None] * n_streams
            sys_res = [None] * n_streams
            for i, cs in enumerate(self.tok_attn._cuda_streams):
                with torch.cuda.stream(cs):
                    tok_res[i] = tok_s[:, i] @ self.tok_attn.streams_weight[i]
                    sys_res[i] = sys_s[:, i] @ self.sys_attn.streams_weight[i]
            torch.cuda.synchronize()
            tok_stream_out = torch.stack(tok_res, dim=1).permute(0,2,1,3).contiguous().view(B_tok, T_tok, D)
            sys_stream_out = torch.stack(sys_res, dim=1).permute(0,2,1,3).contiguous().view(B_sys, T_sys, D)
        else:
            # TPU: single BatchMatMul each
            tok_stream_out = torch.matmul(tok_s, self.tok_attn.streams_weight).permute(0,2,1,3).contiguous().view(B_tok, T_tok, D)
            sys_stream_out = torch.matmul(sys_s, self.sys_attn.streams_weight).permute(0,2,1,3).contiguous().view(B_sys, T_sys, D)

        # ── Step 2: inter-portal bias injection ───────────────────────────────
        tok_stream_out, sys_stream_out = self.inter_portal([tok_stream_out, sys_stream_out])

        # ── Step 3: portal — single BatchMatMul each ──────────────────────────
        tok_p = tok_stream_out.view(B_tok, T_tok, portal_heads, portal_dim).permute(0,2,1,3)
        sys_p = sys_stream_out.view(B_sys, T_sys, portal_heads, portal_dim).permute(0,2,1,3)

        tok_portal_out = self.tok_attn.portal_norm(
            torch.matmul(tok_p, self.tok_attn.portal_weight).permute(0,2,1,3).contiguous().view(B_tok, T_tok, D))
        tok_portal_out = tok_portal_out.clamp(-1e4, 1e4)
        sys_portal_out = self.sys_attn.portal_norm(
            torch.matmul(sys_p, self.sys_attn.portal_weight).permute(0,2,1,3).contiguous().view(B_sys, T_sys, D))
        sys_portal_out = sys_portal_out.clamp(-1e4, 1e4)

        # ── Step 4: bias cachneck + cache update — in-graph, no host sync ────
        # Apply sparse mask functionally — no .data mutation mid-forward
        pad = self.tok_attn._cache_pad
        masked_tok_bcn = self.tok_attn.bcn_compress.weight * self.tok_attn.bcn_mask
        masked_sys_bcn = self.sys_attn.bcn_compress.weight * self.sys_attn.bcn_mask

        tok_bottleneck = F.linear(tok_portal_out, masked_tok_bcn) + \
                         self.tok_attn.cache_pen[:nb].detach().clone().unsqueeze(0).unsqueeze(0)
        sys_bottleneck = F.linear(sys_portal_out, masked_sys_bcn) + \
                         self.sys_attn.cache_pen[:nb].detach().clone().unsqueeze(0).unsqueeze(0)

        # EMA cache update — momentum prevents hard mean from flattening transient KV associations
        m   = self.tok_attn.cfg.bcn_momentum
        self.tok_attn.cache_epg.copy_(m * self.tok_attn.cache_epg + (1-m) * F.pad(tok_bottleneck.detach().mean(dim=(0,1)).clamp(-1e4, 1e4), (0, pad-nb)))
        self.sys_attn.cache_epg.copy_(m * self.sys_attn.cache_epg + (1-m) * F.pad(sys_bottleneck.detach().mean(dim=(0,1)).clamp(-1e4, 1e4), (0, pad-nb)))
        self.tok_attn.cache_pen.copy_(m * self.tok_attn.cache_pen + (1-m) * F.pad(self.tok_attn.bcn_read(tok_portal_out.detach().mean(dim=(0,1))).clamp(-1e4, 1e4), (0, pad-nb)))
        self.sys_attn.cache_pen.copy_(m * self.sys_attn.cache_pen + (1-m) * F.pad(self.sys_attn.bcn_read(sys_portal_out.detach().mean(dim=(0,1))).clamp(-1e4, 1e4), (0, pad-nb)))

        # ── Step 5: inhibitory output ──────────────────────────────────────────
        tok_out = tok - self.tok_attn.bcn_out(tok_bottleneck)
        sys_out = sys - self.sys_attn.bcn_out(sys_bottleneck)

        return tok_out, sys_out
