"""Official FlashAttention blocks with the G1 head-wise output gate."""

from functools import partial

import torch
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.modules.mha import MHA


class VarlenRotaryMHA(MHA):
    """Connect FlashAttention's varlen RoPE kernel to its official MHA module."""

    def forward(self, x, *args, **kwargs):
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = kwargs.get("max_seqlen")
        if cu_seqlens is None or self.rotary_emb_dim == 0:
            return super().forward(x, *args, **kwargs)
        if args or self.cross_attn or kwargs.get("x_kv") is not None:
            raise ValueError("Packed RoPE adapter only supports self-attention")
        if max_seqlen is None:
            raise ValueError("max_seqlen is required for packed RoPE")

        qkv = rearrange(
            self.Wqkv(x),
            "t (three h d) -> t three h d",
            three=3,
            d=self.head_dim,
        )
        self.rotary_emb._update_cos_sin_cache(
            max_seqlen,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        rotary_kwargs = {
            "cos": self.rotary_emb._cos_cached,
            "sin": self.rotary_emb._sin_cached,
            "interleaved": self.rotary_emb.interleaved,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        # Q and K use identical positions, so rotate their 2H heads in one
        # official varlen kernel instead of launching one kernel per tensor.
        query_key = qkv.flatten(1, 2)[:, : 2 * self.num_heads]
        query_key = apply_rotary_emb(query_key, **rotary_kwargs)
        query_key = query_key.view(-1, 2, self.num_heads, self.head_dim)
        qkv = torch.cat((query_key, qkv[:, 2:]), dim=1)
        context = self.inner_attn(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return self.out_proj(rearrange(context, "... h d -> ... (h d)"))


class HeadwiseGatedMHA(VarlenRotaryMHA):
    """Apply a query-dependent scalar gate to each SDPA head output."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        self.gate_proj = torch.nn.Linear(embed_dim, num_heads, bias=False)
        self._active_gate = None
        self.out_proj.register_forward_pre_hook(self._gate_out_projection_input)

    def _gate_out_projection_input(self, _module, args):
        if self._active_gate is None:
            raise RuntimeError("G1 output projection hook called outside attention forward")
        return (args[0] * self._active_gate, *args[1:])

    def forward(self, x, *args, **kwargs):
        gate = torch.sigmoid(self.gate_proj(x))
        self._active_gate = (
            gate.unsqueeze(-1)
            .expand(*gate.shape, self.head_dim)
            .reshape(*gate.shape[:-1], self.embed_dim)
        )
        try:
            return super().forward(x, *args, **kwargs)
        finally:
            self._active_gate = None


def make_flash_block(
    *,
    dim,
    num_heads,
    hidden_dim,
    causal,
    gated,
    use_flash_attn=True,
    cross_attn=False,
    rotary=True,
    dropout=0.0,
    layer_idx=None,
    with_mlp=True,
    prenorm=True,
):
    """Construct the official Dao-AILab Block around an official MHA mixer."""
    from flash_attn.modules.block import Block
    from flash_attn.modules.mlp import Mlp

    attention_cls = HeadwiseGatedMHA if gated else VarlenRotaryMHA
    mixer_cls = partial(
        attention_cls,
        num_heads=num_heads,
        causal=causal,
        cross_attn=cross_attn,
        rotary_emb_dim=dim // num_heads if rotary and not cross_attn else 0,
        use_flash_attn=use_flash_attn,
        dropout=dropout,
        layer_idx=layer_idx,
    )
    mlp_cls = partial(Mlp, hidden_features=hidden_dim) if with_mlp else torch.nn.Identity
    return Block(
        dim,
        mixer_cls=mixer_cls,
        mlp_cls=mlp_cls,
        prenorm=prenorm,
        resid_dropout1=dropout,
        resid_dropout2=dropout,
        fused_dropout_add_ln=use_flash_attn,
        residual_in_fp32=prenorm,
    )
