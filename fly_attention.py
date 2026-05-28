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
    sparsity: float = 0.5     # fraction of Δ7 weights zeroed (biological sparse connectivity)
    portal_heads: int = 4     # topographic portal — BU/LAL convergence heads
    kc_size: int = -1         # Kenyon cell expansion size (-1 = auto: 2×d_model, 0 = disabled)
                               # expansion layer before bcn — sparse hash of portal activations
                               # functional: wider → more discriminable sparse codes
                               # biological: MB Kenyon cells expand 51 glomeruli → 2000 KCs
                               # simplified: just expand + GELU + top-k (≈APL global inhibition)
    ring_alpha: float = 0.1   # ring attractor conv strength (0=disabled)
                               # biological: EB ring attractor — local excitation stabilizes cache bump



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

        # ── Parallel streams (PB columns) — fixed random orthogonal ─────────────
        # Biologically: PB columns are genetically hardwired topographic maps.
        # Fixed across all flies — no learning, no plasticity.
        # Each column sees only its own slice of d_model (block-diagonal structure).
        # Fixed orthogonal init — preserves signal magnitude, no gradient needed.
        # Registered as buffer not Parameter — no gradient, no optimizer state.
        # Mobile export: weights.unbind(0) → 16 separate tensors for NPU inference
        streams_blocks = []
        for _ in range(config.n_streams):
            b = torch.empty(self.stream_dim, self.stream_dim)
            nn.init.orthogonal_(b)
            streams_blocks.append(b)
        self.register_buffer('streams_weight', torch.stack(streams_blocks))

        # ── Portal / BU-LAL — fixed structured orthogonal ───────────────────────
        # Biologically: BU-LAL convergence is anatomically fixed — highly ordered
        # topographic convergence, same across all flies (194 distinct PB neuron types).
        # Fixed orthogonal init — preserves norm, structured not random.
        # Registered as buffer — no gradient, no optimizer state, O(1) memory.
        # portal_norm IS learned — the scaling/shifting of the convergence output
        # is the one place the portal can adapt (like Hebbian weight scaling).
        portal_blocks = []
        for _ in range(config.portal_heads):
            b = torch.empty(self.portal_dim, self.portal_dim)
            nn.init.orthogonal_(b)
            portal_blocks.append(b)
        self.register_buffer('portal_weight', torch.stack(portal_blocks))
        self.portal_norm = nn.LayerNorm(config.d_model)  # learned — fine-tunes convergence

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
        # bcn_compress input is KC output if KC enabled, else portal_out
        _kc_size = config.kc_size if config.kc_size >= 0 else config.d_model * 2
        _bcn_in = _kc_size if _kc_size > 0 else config.d_model
        self.bcn_compress = nn.Linear(_bcn_in, config.bcn_size, bias=False)
        mask = torch.zeros(config.bcn_size, _bcn_in)
        n_connections = int(_bcn_in * (1.0 - config.sparsity))
        for i in range(config.bcn_size):
            idx = torch.randperm(_bcn_in)[:n_connections]
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

        # ── bcn_gate: learned write gate — how strongly to update cache ───────────
        # sigmoid(gate) near 0.5 at init → balanced read/write
        # model learns: high confidence step → gate→1 (write strongly)
        #               low confidence step  → gate→0 (preserve existing state)
        # biological analogy: dopaminergic gating
        self.bcn_gate = nn.Linear(config.bcn_size, config.bcn_size, bias=True)
        nn.init.zeros_(self.bcn_gate.weight)
        nn.init.constant_(self.bcn_gate.bias, -1.0)  # start conservative — gate≈0.27, opens as surprise grows

        # ── KC sparse expansion layer (MB Kenyon cells) ──────────────────────
        # Expands portal representation → sparse hash via GELU + top-k
        # GELU approximates the threshold nonlinearity of KC neurons
        # top-k approximates APL global inhibition (one GABAergic neuron inhibits all KCs)
        # Functional purpose: wider expansion → more discriminable sparse codes
        # Biological: 51 glomeruli → 2000 KCs (40x expansion), ~5% active at once
        # Our default: 2x expansion — enough discriminability, not too expensive
        if _kc_size > 0:
            # Fixed random sparse projection — biologically accurate:
            # KC connectivity is random (confirmed by FlyWire connectome) but FIXED at birth
            # Each KC connects to only ~6 random projection neurons (1-7 range)
            # n_inputs_per_kc ≈ d_model * 6/n_projection_neurons ≈ 3-5% of d_model
            n_inputs_per_kc = max(1, config.d_model // 20)  # ~5% sparse input per KC
            kc_w = torch.zeros(_kc_size, config.d_model)
            for i in range(_kc_size):
                idx = torch.randperm(config.d_model)[:n_inputs_per_kc]
                # random normal weights — not all equal strength (biological variation)
                kc_w[i, idx] = torch.randn(n_inputs_per_kc) * (1.0 / n_inputs_per_kc) ** 0.5
            self.register_buffer('kc_weight', kc_w)  # fixed, not learned
            self._kc_size = _kc_size
        else:
            self.kc_weight = None
            self._kc_size = 0

        # ── Ring attractor conv (EB ring attractor) ──────────────────────────
        # Small learned 1D circular conv — fixed topology, tiny learnable weights.
        # Biological: EPG→EPG recurrent connections are topologically fixed (ring),
        # but synapse strengths are finely tuned by evolution (not random).
        # Center-surround init: local excitation + surround inhibition = bump dynamics.
        # Only 5 parameters — can adapt the bump shape slightly for generalization.
        # ring_alpha=0 → disabled, >0 → scales conv output before adding to cache.
        self.ring_conv = nn.Conv1d(1, 1, kernel_size=5, padding=0, bias=False)
        nn.init.constant_(self.ring_conv.weight, 0.0)
        # center-surround init — excite center, inhibit surround
        with torch.no_grad():
            self.ring_conv.weight[0, 0] = torch.tensor([-0.1, 0.3, 1.0, 0.3, -0.1])

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
        # set _suppress_training_detach=True for multi-step diffusion training
        # where cache gradient SHOULD flow between steps
        if self.training and not getattr(self, '_suppress_training_detach', False):
            self.cache_epg.detach_()
            self.cache_pen.detach_()

        # ── Step 1: Parallel streams — single BatchMatMul ─────────────────────
        # EPG cache bias: only use first nb slots, rest is padding
        epg_bias = self.bcn_expand(self.cache_epg[:nb].clone())
        x_biased = x + epg_bias.unsqueeze(0).unsqueeze(0)

        # reshape for batched matmul: (B, T, d_model) → (B, n_streams, T, stream_dim)
        x_streams = x_biased.view(B, T, self.cfg.n_streams, self.stream_dim)
        x_streams = x_streams.permute(0, 2, 1, 3)  # (B, n_streams, T, stream_dim)

        # single BatchMatMul on all hardware — streams_weight is a fixed buffer (no grad)
        # at stream_dim=48, CUDA stream dispatch overhead exceeds parallelism benefit
        # one batched matmul is faster than 16 separately dispatched tiny matmuls
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

        # ── Step 3a: KC sparse expansion (MB Kenyon cells) ───────────────────
        # expand → GELU → top-k sparse → sparse hash of portal activations
        # GELU ≈ KC threshold nonlinearity
        # top-k ≈ APL global inhibition (keeps ~5% active)
        # result: highly discriminable sparse code fed into bcn_compress
        if self.kc_weight is not None:
            # fixed random sparse expansion — no gradient through kc_weight
            kc = F.gelu(F.linear(portal_out, self.kc_weight))       # (B, T, kc_size)
            # APL global inhibition — single GABAergic neuron inhibits all KCs uniformly
            # APL fires proportional to mean activity → only strongest KCs survive
            # O(n) vs top-k O(n log n), differentiable, compiler-friendly
            apl = kc.mean(dim=-1, keepdim=True)                      # mean field inhibition
            portal_for_bcn = F.relu(kc - apl)                        # winners survive
        else:
            portal_for_bcn = portal_out

        # ── Step 3b: PFL multiplicative modulation ────────────────────────────
        # PFL neurons: two dendrites multiply heading × goal → error signal
        # Here: portal × cache_expansion → "how does current input relate to cached state"
        # Hadamard gate — cache state modulates what portal pays attention to
        # biological: PFL1/2/3 coordinate transform, zero parameters
        cache_bias = self.bcn_expand(self.cache_epg[:nb].clone())          # (d_model,) — clone needed: buffer modified later in same forward
        portal_out = portal_out * (1.0 + cache_bias.unsqueeze(0).unsqueeze(0))

        # ── Step 3c: bias cachneck compression ───────────────────────────────
        # Apply sparse mask functionally — avoids .data mutation mid-forward
        masked_bcn = self.bcn_compress.weight * self.bcn_mask
        bottleneck = F.linear(portal_for_bcn, masked_bcn) +                      self.cache_pen[:nb].detach().unsqueeze(0).unsqueeze(0)

        # ── Step 4: Cache update — velocity + surprise + ring attractor ───────
        # Full biological stack:
        #   velocity  = P-EN angular velocity neurons (rate of change)
        #   surprise  = OA/dopamine prediction error (how unexpected is this?)
        #   gate      = dopaminergic write gate driven by surprise
        #   ring_conv = EB ring attractor (local excitation → stable bump)
        cache_prev = self.cache_epg.detach().clone()

        with torch.no_grad():
            bcn_mean = bottleneck.mean(dim=(0, 1)).clamp(-1e4, 1e4)  # (nb,)
            velocity = bcn_mean - cache_prev[:nb]                     # P-EN: rate of change
            surprise = velocity.abs().mean().clamp(0, 10)             # OA: prediction error
            gate     = torch.sigmoid(
                self.bcn_gate(bcn_mean) + 0.1 * surprise                    # surprise-driven gate
            )                                                          # (nb,) in (0,1)
            # integrate velocity into cache value (P-EN bump shift)
            new_val  = (bcn_mean + 0.1 * velocity).clamp(-1e4, 1e4)  # (nb,)
            # ring attractor — learned circular conv stabilizes cache into bump
            if self.cfg.ring_alpha > 0:
                v_conv = new_val.view(1, 1, nb)                       # (1,1,nb)
                v_conv = F.pad(v_conv, (2, 2), mode='circular')       # circular padding
                v_conv = self.ring_conv(v_conv)                        # (1,1,nb) learned conv
                new_val = new_val + self.cfg.ring_alpha * v_conv.squeeze()

        new_epg  = F.pad(new_val, (0, self._cache_pad - nb))
        gate_pad = F.pad(gate,    (0, self._cache_pad - nb))
        self.cache_epg.copy_(gate_pad * new_epg + (1 - gate_pad) * self.cache_epg)

        raw_pen  = self.bcn_read(portal_out.detach().mean(dim=(0, 1))).clamp(-1e4, 1e4)
        new_pen  = F.pad(raw_pen, (0, self._cache_pad - nb))
        self.cache_pen.copy_(gate_pad * new_pen + (1 - gate_pad) * self.cache_pen)

        # ── Step 5: Inhibitory output ─────────────────────────────────────────
        # Subtracts what's established in cache from input → passes novel signal
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

        # kc_weight is a fixed buffer — no gradient, no hook needed

    def param_count(self) -> dict[str, int]:
        """Breakdown of parameter counts per component."""
        return {
            'streams':      self.streams_weight.numel(),
            'portal':       self.portal_weight.numel(),
            'bcn_compress':       self.bcn_compress.weight.numel(),
            'bcn_expand':   self.bcn_expand.weight.numel(),
            'bcn_read':     self.bcn_read.weight.numel(),
            'bcn_out':      self.bcn_out.weight.numel(),
            'kc_weight':    self.kc_weight.numel() if self.kc_weight is not None else 0,  # fixed buffer
            'ring_conv':    self.ring_conv.weight.numel(),  # 5 learned params',
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
        tok_epg_bias = self.tok_attn.bcn_expand(self.tok_attn.cache_epg[:nb].clone())
        sys_epg_bias = self.sys_attn.bcn_expand(self.sys_attn.cache_epg[:nb].clone())

        tok_biased = tok + tok_epg_bias.unsqueeze(0).unsqueeze(0)
        sys_biased = sys + sys_epg_bias.unsqueeze(0).unsqueeze(0)

        # reshape: (B, T, d) → (B, n_streams, T, stream_dim)
        tok_s = tok_biased.view(B_tok, T_tok, n_streams, stream_dim).permute(0, 2, 1, 3)
        sys_s = sys_biased.view(B_sys, T_sys, n_streams, stream_dim).permute(0, 2, 1, 3)

        # single BatchMatMul on all hardware — no synchronize() overhead
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

        # ── Step 4: KC expansion + PFL modulation + bias cachneck ────────────
        pad = self.tok_attn._cache_pad

        # KC sparse expansion (MB Kenyon cells) — same as FlyAttention.forward Step 3a
        def _kc_expand(attn, portal):
            if attn.kc_weight is not None:
                kc = F.gelu(F.linear(portal, attn.kc_weight))
                apl = kc.mean(dim=-1, keepdim=True)   # APL global inhibition
                return F.relu(kc - apl)                # winners survive
            return portal

        tok_for_bcn = _kc_expand(self.tok_attn, tok_portal_out)
        sys_for_bcn = _kc_expand(self.sys_attn, sys_portal_out)

        # PFL multiplicative modulation — cache gates what portal attends to
        tok_portal_out = tok_portal_out * (1.0 + tok_epg_bias.unsqueeze(0).unsqueeze(0))
        sys_portal_out = sys_portal_out * (1.0 + sys_epg_bias.unsqueeze(0).unsqueeze(0))

        # Apply sparse mask functionally — no .data mutation mid-forward
        masked_tok_bcn = self.tok_attn.bcn_compress.weight * self.tok_attn.bcn_mask
        masked_sys_bcn = self.sys_attn.bcn_compress.weight * self.sys_attn.bcn_mask

        tok_bottleneck = F.linear(tok_for_bcn, masked_tok_bcn) + \
                         self.tok_attn.cache_pen[:nb].detach().unsqueeze(0).unsqueeze(0)
        sys_bottleneck = F.linear(sys_for_bcn, masked_sys_bcn) + \
                         self.sys_attn.cache_pen[:nb].detach().unsqueeze(0).unsqueeze(0)

        # velocity + surprise + ring attractor cache update (matches FlyAttention.forward)
        with torch.no_grad():
            tok_mean = tok_bottleneck.detach().mean(dim=(0,1)).clamp(-1e4,1e4)
            sys_mean = sys_bottleneck.detach().mean(dim=(0,1)).clamp(-1e4,1e4)
            # P-EN velocity — rate of change
            tok_vel  = tok_mean - self.tok_attn.cache_epg[:nb]
            sys_vel  = sys_mean - self.sys_attn.cache_epg[:nb]
            # OA surprise — prediction error drives gate
            tok_surp = tok_vel.abs().mean().clamp(0, 10)
            sys_surp = sys_vel.abs().mean().clamp(0, 10)
            tok_gate = F.pad(torch.sigmoid(self.tok_attn.bcn_gate(tok_mean) + 0.1 * tok_surp), (0, pad-nb))
            sys_gate = F.pad(torch.sigmoid(self.sys_attn.bcn_gate(sys_mean) + 0.1 * sys_surp), (0, pad-nb))
            # integrate velocity
            tok_new  = (tok_mean + 0.1 * tok_vel).clamp(-1e4,1e4)
            sys_new  = (sys_mean + 0.1 * sys_vel).clamp(-1e4,1e4)
            # ring attractor conv
            ra = self.tok_attn.cfg.ring_alpha if hasattr(self.tok_attn.cfg, 'ring_alpha') else 0.1
            if ra > 0:
                def _ring(v, attn):
                    vc = F.pad(v.view(1,1,-1), (2,2), mode='circular')
                    return v + ra * attn.ring_conv(vc).squeeze()
                tok_new = _ring(tok_new, self.tok_attn)
                sys_new = _ring(sys_new, self.sys_attn)

        self.tok_attn.cache_epg.copy_(tok_gate * F.pad(tok_new,(0,pad-nb)) + (1-tok_gate) * self.tok_attn.cache_epg)
        self.sys_attn.cache_epg.copy_(sys_gate * F.pad(sys_new,(0,pad-nb)) + (1-sys_gate) * self.sys_attn.cache_epg)
        self.tok_attn.cache_pen.copy_(tok_gate * F.pad(self.tok_attn.bcn_read(tok_portal_out.detach().mean(dim=(0,1))).clamp(-1e4,1e4),(0,pad-nb)) + (1-tok_gate) * self.tok_attn.cache_pen)
        self.sys_attn.cache_pen.copy_(sys_gate * F.pad(self.sys_attn.bcn_read(sys_portal_out.detach().mean(dim=(0,1))).clamp(-1e4,1e4),(0,pad-nb)) + (1-sys_gate) * self.sys_attn.cache_pen)

        # ── Step 5: inhibitory output ──────────────────────────────────────────
        tok_out = tok - self.tok_attn.bcn_out(tok_bottleneck)
        sys_out = sys - self.sys_attn.bcn_out(sys_bottleneck)

        return tok_out, sys_out
