# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V3.2-Exp: DeepSeek-V3 plus DeepSeek Sparse Attention (DSA).

This is DeepSeek-V3 with a lightning indexer added to each attention layer: the indexer scores every
query against the cached keys and keeps the top-`index_topk` tokens, which become an additive sparse
mask folded into the MLA attention mask. Everything else (MoE, MLA projections, RoPE, the decoder /
model / causal-LM scaffolding) is inherited unchanged from DeepSeek-V3.

The cross-layer top-k *sharing* variant is a GLM-MoE-DSA innovation and lives in that model, which
inherits from this one (see `models/glm_moe_dsa/modular_glm_moe_dsa.py`).
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer
from .sparse_mla_triton import sparse_mla_triton, sparse_mla_triton_projected, sparse_mla_triton_valid_all


logger = logging.get_logger(__name__)


def deepseek_mla_attention_forward(*args, **kwargs):
    raise RuntimeError("DeepSeek-V3.2 handles `deepseek_mla` inside `DeepseekV32Attention.forward`.")


def deepseek_mla_triton_attention_forward(*args, **kwargs):
    raise RuntimeError("DeepSeek-V3.2 handles `deepseek_mla_triton` inside `DeepseekV32Attention.forward`.")


ALL_ATTENTION_FUNCTIONS.register("deepseek_mla", deepseek_mla_attention_forward)
ALL_ATTENTION_FUNCTIONS.register("deepseek_mla_triton", deepseek_mla_triton_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("deepseek_mla", sdpa_mask)
ALL_MASK_ATTENTION_FUNCTIONS.register("deepseek_mla_triton", sdpa_mask)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V3.2-Exp")
@strict
class DeepseekV32Config(Glm4MoeLiteConfig, RotaryEmbeddingConfigMixin):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of leading layers that use a dense MLP; the rest use the MoE block.

    ```python
    >>> from transformers import DeepseekV32Config, DeepseekV32Model

    >>> # Initializing a DeepSeek-V3.2 configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    mlp_bias: bool = False
    head_dim: int = 64
    first_k_dense_replace: int = 3
    pretraining_tp = AttributeError()
    rope_interleave = AttributeError()
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so point `head_dim` at it: the inherited (Llama) rotary
        # embedding reads `config.head_dim` and then computes the right frequencies with no override needed.
        self.head_dim = self.qk_rope_head_dim
        # MLP layer types: the first `first_k_dense_replace` layers are dense, the rest are MoE.
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense)
        # Every layer is DSA — drives cache-class dispatch.
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        # BC: re-route `num_experts` to `n_routed_experts`
        if (num_experts := kwargs.get("num_experts")) is not None:
            self.n_routed_experts = num_experts

        super().__post_init__(**kwargs)


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV32Indexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer for selecting top-k tokens.

    The Indexer has its own lightweight projections (wq_b, wk) separate from the main MLA attention,
    and returns the additive top-k sparse mask directly (`0` at the selected tokens, `-inf` elsewhere);
    the raw top-k indices are only ever scattered into that mask, so they are not surfaced.

    **Cache strategy**: the indexer key cache lives on the per-layer `DynamicIndexedLayer` (or the
    `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config: "DeepseekV32Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        This is the bf16 equivalent of the reference Indexer which uses `rotate_activation` (Hadamard transform)
        and `fp8_index` (FP8 quantized scoring kernel). Since the Hadamard transform is orthogonal (dot products
        are preserved: Hq·Hk = q·k), and FP8 quantization is a precision optimization, we skip both and compute
        scores directly in bf16/fp32.

        The scoring logic computes:
            index_score[b,s,t] = Σ_h (weight[b,s,h] · softmax_scale · q[b,s,h,:] · k[b,t,:])

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                turn these into an additive sparse mask; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # The indexer uses NON-interleaved (half-split) RoPE — unlike the main MLA attention
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask is not None:
            index_scores = index_scores + attention_mask
        else:
            key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
            causal = key_positions[None, None, :] > position_ids[:, :, None]  # [B, S, T]
            index_scores = index_scores.masked_fill(causal, float("-inf"))

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class DeepseekV32Attention(DeepseekV3Attention):
    """
    DeepSeek-V3 MLA, with a DSA indexer whose top-k sparse mask is folded into the attention mask.
    Qlora rank formulation is dropped as it is never used in released models.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        use_flash_attention = self.config._attn_implementation == "flash_attention_2"
        use_deepseek_mla = self.config._attn_implementation == "deepseek_mla"
        use_deepseek_mla_triton = self.config._attn_implementation == "deepseek_mla_triton"

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        sparse_indices = None
        if use_deepseek_mla or use_deepseek_mla_triton:
            if past_key_values is not None:
                raise NotImplementedError(
                    "DeepSeek-V3.2 MLA attention currently supports training/prefill without cache."
                )

            kv_b_proj_weight = self.kv_b_proj.weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            w_uk = kv_b_proj_weight[:, : self.qk_nope_head_dim, :]
            w_uv = kv_b_proj_weight[:, self.qk_nope_head_dim :, :]

            key_positions = torch.arange(seq_length, device=hidden_states.device)
            key_states = torch.cat((kv_c_normed[:, :, None, :], k_rot.transpose(1, 2)), dim=-1)
            value_states = F.pad(kv_c_normed[:, :, None, :], (0, self.qk_rope_head_dim))
            _, _, num_key_value_heads, qk_head_dim = key_states.shape
            key_states = key_states.squeeze(2)
            value_states = value_states.squeeze(2)

            indexer_k = self.indexer.k_norm(self.indexer.wk(hidden_states)).unsqueeze(2)
            indexer_k_rot, indexer_k_pass = torch.split(
                indexer_k,
                [self.indexer.qk_rope_head_dim, self.indexer.head_dim - self.indexer.qk_rope_head_dim],
                dim=-1,
            )
            _, indexer_k_rot = apply_rotary_pos_emb(
                indexer_k_rot, indexer_k_rot, position_embeddings[0], position_embeddings[1], unsqueeze_dim=2
            )
            indexer_k = torch.cat([indexer_k_rot, indexer_k_pass], dim=-1).squeeze(2)

            if use_deepseek_mla:
                from flash_attn import flash_attn_varlen_func

            attn_output = hidden_states.new_empty(batch_size, seq_length, self.num_heads, self.v_head_dim)
            chunk_size = getattr(self.config, "dsa_chunk_size", 128)
            for chunk_start in range(0, seq_length, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_length)
                chunk_len = chunk_end - chunk_start

                ql_nope = torch.einsum("bhsp,hpl->bshl", q_pass[:, :, chunk_start:chunk_end], w_uk)
                query_states = torch.cat((ql_nope, q_rot[:, :, chunk_start:chunk_end].transpose(1, 2)), dim=-1)

                indexer_q = self.indexer.wq_b(q_resid[:, chunk_start:chunk_end])
                indexer_q = indexer_q.view(batch_size, chunk_len, self.indexer.n_heads, self.indexer.head_dim)
                indexer_q_rot, indexer_q_pass = torch.split(
                    indexer_q,
                    [self.indexer.qk_rope_head_dim, self.indexer.head_dim - self.indexer.qk_rope_head_dim],
                    dim=-1,
                )
                indexer_q_rot, _ = apply_rotary_pos_emb(
                    indexer_q_rot,
                    indexer_k_rot[:, chunk_start:chunk_end],
                    position_embeddings[0][:, chunk_start:chunk_end],
                    position_embeddings[1][:, chunk_start:chunk_end],
                    unsqueeze_dim=2,
                )
                indexer_q = torch.cat([indexer_q_rot, indexer_q_pass], dim=-1)
                indexer_scores = (
                    torch.matmul(indexer_q.float(), indexer_k.transpose(-1, -2).float().unsqueeze(1))
                    * self.indexer.softmax_scale
                )
                indexer_scores = F.relu(indexer_scores)
                indexer_weights = self.indexer.weights_proj(
                    hidden_states[:, chunk_start:chunk_end].to(self.indexer.weights_proj.weight.dtype)
                ).float() * (self.indexer.n_heads**-0.5)
                index_scores = torch.matmul(indexer_weights.unsqueeze(-2), indexer_scores).squeeze(-2)

                valid_indexer_positions = key_positions[None, None, :] <= position_ids[:, chunk_start:chunk_end, None]
                valid_indexer_positions = valid_indexer_positions.expand(batch_size, -1, -1)
                if attention_mask is not None:
                    if attention_mask.ndim == 4:
                        valid_indexer_positions = (
                            valid_indexer_positions & attention_mask[:, 0, chunk_start:chunk_end, :].bool()
                        )
                    else:
                        valid_indexer_positions = (
                            valid_indexer_positions & attention_mask[:, None, -seq_length:].bool()
                        )
                index_scores = index_scores.masked_fill(~valid_indexer_positions, float("-inf"))
                topk = min(self.indexer.index_topk, seq_length)
                topk_indices = index_scores.topk(topk, dim=-1).indices.to(torch.int32)
                valid_topk = None

                if use_deepseek_mla_triton:
                    topk_sort_order = topk_indices.argsort(dim=-1)
                    topk_indices = topk_indices.gather(-1, topk_sort_order)
                    if self.training and torch.is_grad_enabled():
                        chunk_valid_all = attention_mask is None and bool(
                            torch.all(position_ids[:, chunk_start:chunk_end] >= topk - 1)
                        )
                        if not chunk_valid_all:
                            valid_topk = valid_indexer_positions.gather(-1, topk_indices.long())
                            chunk_valid_all = bool(torch.all(valid_topk))
                        sparse_mla_fn = sparse_mla_triton_valid_all if chunk_valid_all else sparse_mla_triton
                        latent_output = sparse_mla_fn(
                            query_states,
                            key_states,
                            value_states,
                            topk_indices,
                            topk_indices if chunk_valid_all else valid_topk,
                            self.scaling,
                        )[..., : self.kv_lora_rank]
                        attn_output[:, chunk_start:chunk_end] = torch.einsum("bshl,hvl->bshv", latent_output, w_uv)
                    else:
                        valid_topk = valid_indexer_positions.gather(-1, topk_indices.long())
                        attn_output[:, chunk_start:chunk_end] = sparse_mla_triton_projected(
                            query_states,
                            key_states,
                            value_states,
                            topk_indices,
                            valid_topk,
                            w_uv,
                            self.scaling,
                        )
                else:
                    valid_topk = valid_indexer_positions.gather(-1, topk_indices.long())
                    key_gather_indices = (
                        topk_indices[:, :, :, None, None].long().expand(batch_size, chunk_len, topk, 1, qk_head_dim)
                    )
                    selected_key_states = (
                        key_states[:, None, :, None, :]
                        .expand(batch_size, chunk_len, seq_length, 1, qk_head_dim)
                        .gather(2, key_gather_indices)
                        .reshape(batch_size * chunk_len, topk, 1, qk_head_dim)
                    )
                    selected_value_states = (
                        value_states[:, None, :, None, :]
                        .expand(batch_size, chunk_len, seq_length, 1, qk_head_dim)
                        .gather(2, key_gather_indices)
                        .reshape(batch_size * chunk_len, topk, 1, qk_head_dim)
                    )
                    valid_topk_flat = valid_topk.reshape(batch_size * chunk_len, topk)
                    key_lengths = valid_topk_flat.sum(dim=-1, dtype=torch.int32)
                    cu_seqlens_q = torch.arange(
                        batch_size * chunk_len + 1, device=hidden_states.device, dtype=torch.int32
                    )
                    cu_seqlens_k = F.pad(torch.cumsum(key_lengths, dim=0, dtype=torch.int32), (1, 0))
                    latent_output = flash_attn_varlen_func(
                        query_states.reshape(batch_size * chunk_len, self.num_heads, qk_head_dim),
                        selected_key_states[valid_topk_flat],
                        selected_value_states[valid_topk_flat],
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=1,
                        max_seqlen_k=int(key_lengths.max().item()),
                        dropout_p=0.0 if not self.training else self.attention_dropout,
                        softmax_scale=self.scaling,
                        causal=False,
                    ).view(batch_size, chunk_len, self.num_heads, qk_head_dim)[..., : self.kv_lora_rank]
                    attn_output[:, chunk_start:chunk_end] = torch.einsum("bshl,hvl->bshv", latent_output, w_uv)
            attn_weights = None
        else:
            k_pass = self.kv_b_proj(kv_c_normed).view(key_shape).transpose(1, 2)
            k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

            query_states = torch.cat((q_pass, q_rot), dim=-1)
            key_states = torch.cat((k_pass, k_rot), dim=-1)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if use_deepseek_mla or use_deepseek_mla_triton:
            pass
        elif use_flash_attention:
            value_head_dim = value_states.shape[-1]
            if value_head_dim != query_states.shape[-1]:
                value_states = F.pad(value_states, (0, query_states.shape[-1] - value_head_dim))

            _, num_heads, key_length, qk_head_dim = key_states.shape
            key_positions = torch.arange(key_length, device=hidden_states.device)
            indexer_k = self.indexer.k_norm(self.indexer.wk(hidden_states)).unsqueeze(2)
            indexer_k_rot, indexer_k_pass = torch.split(
                indexer_k,
                [self.indexer.qk_rope_head_dim, self.indexer.head_dim - self.indexer.qk_rope_head_dim],
                dim=-1,
            )
            _, indexer_k_rot = apply_rotary_pos_emb(
                indexer_k_rot, indexer_k_rot, position_embeddings[0], position_embeddings[1], unsqueeze_dim=2
            )
            indexer_k = torch.cat([indexer_k_rot, indexer_k_pass], dim=-1).squeeze(2)

            from flash_attn import flash_attn_varlen_func

            attn_output = hidden_states.new_empty(batch_size, seq_length, num_heads, value_head_dim)
            chunk_size = getattr(self.config, "dsa_chunk_size", 128)
            for chunk_start in range(0, seq_length, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_length)
                chunk_len = chunk_end - chunk_start
                indexer_q = self.indexer.wq_b(q_resid[:, chunk_start:chunk_end])
                indexer_q = indexer_q.view(batch_size, chunk_len, self.indexer.n_heads, self.indexer.head_dim)
                indexer_q_rot, indexer_q_pass = torch.split(
                    indexer_q,
                    [self.indexer.qk_rope_head_dim, self.indexer.head_dim - self.indexer.qk_rope_head_dim],
                    dim=-1,
                )
                indexer_q_rot, _ = apply_rotary_pos_emb(
                    indexer_q_rot,
                    indexer_k_rot[:, chunk_start:chunk_end],
                    position_embeddings[0][:, chunk_start:chunk_end],
                    position_embeddings[1][:, chunk_start:chunk_end],
                    unsqueeze_dim=2,
                )
                indexer_q = torch.cat([indexer_q_rot, indexer_q_pass], dim=-1)
                indexer_scores = (
                    torch.matmul(indexer_q.float(), indexer_k.transpose(-1, -2).float().unsqueeze(1))
                    * self.indexer.softmax_scale
                )
                indexer_scores = F.relu(indexer_scores)
                indexer_weights = self.indexer.weights_proj(
                    hidden_states[:, chunk_start:chunk_end].to(self.indexer.weights_proj.weight.dtype)
                ).float() * (self.indexer.n_heads**-0.5)
                index_scores = torch.matmul(indexer_weights.unsqueeze(-2), indexer_scores).squeeze(-2)

                valid_indexer_positions = key_positions[None, None, :] <= position_ids[:, chunk_start:chunk_end, None]
                valid_indexer_positions = valid_indexer_positions.expand(batch_size, -1, -1)
                if attention_mask is not None:
                    if attention_mask.ndim == 4:
                        valid_indexer_positions = (
                            valid_indexer_positions & attention_mask[:, 0, chunk_start:chunk_end, :].bool()
                        )
                    else:
                        valid_indexer_positions = (
                            valid_indexer_positions & attention_mask[:, None, -key_length:].bool()
                        )
                index_scores = index_scores.masked_fill(~valid_indexer_positions, float("-inf"))
                topk = min(self.indexer.index_topk, key_length)
                topk_indices = index_scores.topk(topk, dim=-1).indices.to(torch.int32)
                valid_topk = valid_indexer_positions.gather(-1, topk_indices.long())

                key_gather_indices = (
                    topk_indices[:, None, :, :, None]
                    .long()
                    .expand(batch_size, num_heads, chunk_len, topk, qk_head_dim)
                )
                selected_key_states = (
                    key_states[:, :, None, :, :]
                    .expand(batch_size, num_heads, chunk_len, key_length, qk_head_dim)
                    .gather(3, key_gather_indices)
                    .permute(0, 2, 3, 1, 4)
                    .reshape(batch_size * chunk_len, topk, num_heads, qk_head_dim)
                )
                selected_value_states = (
                    value_states[:, :, None, :, :]
                    .expand(batch_size, num_heads, chunk_len, key_length, qk_head_dim)
                    .gather(3, key_gather_indices)
                    .permute(0, 2, 3, 1, 4)
                    .reshape(batch_size * chunk_len, topk, num_heads, qk_head_dim)
                )
                valid_topk = valid_topk.reshape(batch_size * chunk_len, topk)
                key_lengths = valid_topk.sum(dim=-1, dtype=torch.int32)
                cu_seqlens_q = torch.arange(batch_size * chunk_len + 1, device=hidden_states.device, dtype=torch.int32)
                cu_seqlens_k = F.pad(torch.cumsum(key_lengths, dim=0, dtype=torch.int32), (1, 0))
                chunk_output = flash_attn_varlen_func(
                    query_states[:, :, chunk_start:chunk_end]
                    .transpose(1, 2)
                    .reshape(batch_size * chunk_len, num_heads, qk_head_dim),
                    selected_key_states[valid_topk],
                    selected_value_states[valid_topk],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=1,
                    max_seqlen_k=int(key_lengths.max().item()),
                    dropout_p=0.0 if not self.training else self.attention_dropout,
                    softmax_scale=self.scaling,
                    causal=False,
                ).view(batch_size, chunk_len, num_heads, qk_head_dim)
                if chunk_output.shape[-1] != value_head_dim:
                    chunk_output = chunk_output[..., :value_head_dim]
                attn_output[:, chunk_start:chunk_end] = chunk_output
            attn_weights = None
        else:
            # The indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`.
            indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                position_ids,
                past_key_values=past_key_values,
            )  # [B, S, topk]

        if use_deepseek_mla or use_deepseek_mla_triton or use_flash_attention:
            pass
        elif self.config._attn_implementation in ("eager", "sdpa"):
            # Boolean mask: `True` at keys *not* selected by the indexer (to be masked out).
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )  # [B, 1, S, T]; True = masked
            if attention_mask is None:
                key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
                index_mask = index_mask | (key_positions[None, None, None, :] > position_ids[:, None, :, None])
                attention_mask = hidden_states.new_zeros((batch_size, 1, seq_length, key_states.shape[2]))
            attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        else:
            sparse_indices = topk_indices

        if not (use_deepseek_mla or use_deepseek_mla_triton or use_flash_attention):
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                indices=sparse_indices,
                **kwargs,
            )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False


class DeepseekV32Model(DeepseekV3Model):
    pass


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
]
