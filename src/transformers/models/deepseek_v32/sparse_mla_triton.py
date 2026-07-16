import functools
import os

import torch
import triton
import triton.language as tl


def _get_sparse_mla_kernel_config(head_dim, topk_n):
    block_d = max(16, triton.next_power_of_2(head_dim))
    if block_d >= 512:
        # Large latent MLA dimensions (e.g. 576 -> BLOCK_D 1024) are register-pressure limited.
        # Keeping the sparse tile small improves occupancy on Hopper/Blackwell-class GPUs.
        block_n = 16
        block_h = 16
        num_warps = 8
        num_stages = 1
    elif topk_n >= 64:
        # Small/medium dimensions benefit from fewer sparse-loop iterations.
        block_n = 32
        block_h = 16
        num_warps = 4
        num_stages = 3
    else:
        block_n = 16
        block_h = 16
        num_warps = 4
        num_stages = 3
    return block_d, block_n, block_h, num_warps, num_stages


def _get_sparse_mla_projected_kernel_config(head_dim, value_dim, topk_n):
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_v = min(64, max(16, triton.next_power_of_2(value_dim)))
    block_l = 64
    block_n = 16
    num_warps = 8 if block_d >= 512 else 4
    num_stages = 1 if block_d >= 512 else 3
    return block_d, block_n, block_l, block_v, num_warps, num_stages


@triton.jit
def _sparse_mla_projected_forward_kernel(
    Q,
    K,
    V,
    W_UV,
    TOPK,
    VALID,
    OUT,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LATENT_D: tl.constexpr,
    VALUE_D: tl.constexpr,
    WUV_STRIDE_H: tl.constexpr,
    WUV_STRIDE_V: tl.constexpr,
    WUV_STRIDE_L: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    v_block = pid % tl.cdiv(VALUE_D, BLOCK_V)
    h = (pid // tl.cdiv(VALUE_D, BLOCK_V)) % H
    q_pos = (pid // (tl.cdiv(VALUE_D, BLOCK_V) * H)) % Q_LEN
    b = pid // (tl.cdiv(VALUE_D, BLOCK_V) * H * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = offs_d < D
    v_mask = offs_v < VALUE_D

    q = tl.load(Q + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        valid = tl.load(
            VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int1)

        k_t = tl.load(
            K + ((b * KV_LEN + idx[None, :]) * D + offs_d[:, None]),
            mask=d_mask[:, None] & valid[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q[None, :], k_t, input_precision="ieee")
        scores = tl.reshape(scores, (BLOCK_N,)) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(valid, tl.exp(scores - m_new_safe), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)

        tile_acc = tl.zeros((BLOCK_V,), tl.float32)
        for l_start in range(0, LATENT_D, BLOCK_L):
            offs_l = l_start + tl.arange(0, BLOCK_L)
            l_mask = offs_l < LATENT_D
            latent_v = tl.load(
                V + ((b * KV_LEN + idx[:, None]) * D + offs_l[None, :]),
                mask=valid[:, None] & l_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            w_uv = tl.load(
                W_UV + h * WUV_STRIDE_H + offs_l[:, None] * WUV_STRIDE_L + offs_v[None, :] * WUV_STRIDE_V,
                mask=l_mask[:, None] & v_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            projected_v = tl.dot(latent_v, w_uv, input_precision="ieee")
            tile_acc += tl.sum(p[:, None] * projected_v, axis=0)

        acc = acc * alpha + tile_acc
        m_i = m_new
        l_i = l_new

    out = acc / tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(OUT + (((b * Q_LEN + q_pos) * H + h) * VALUE_D + offs_v), out, mask=v_mask)


@triton.jit
def _sparse_mla_projected_split_stage1_kernel(
    Q,
    K,
    V,
    W_UV,
    TOPK,
    VALID,
    PART_M,
    PART_L,
    PART_ACC,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LATENT_D: tl.constexpr,
    VALUE_D: tl.constexpr,
    WUV_STRIDE_H: tl.constexpr,
    WUV_STRIDE_V: tl.constexpr,
    WUV_STRIDE_L: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    SPLIT_N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % SPLIT_N
    v_blocks: tl.constexpr = tl.cdiv(VALUE_D, BLOCK_V)
    v_block = (pid // SPLIT_N) % v_blocks
    h = (pid // (SPLIT_N * v_blocks)) % H
    q_pos = (pid // (SPLIT_N * v_blocks * H)) % Q_LEN
    b = pid // (SPLIT_N * v_blocks * H * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    d_mask = offs_d < D
    v_mask = offs_v < VALUE_D
    split_start = split_id * SPLIT_SIZE

    q = tl.load(Q + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)

    for rel_start in range(0, SPLIT_SIZE, BLOCK_N):
        offs_n = split_start + rel_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        valid = tl.load(
            VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int1)
        valid = valid & n_mask

        k_t = tl.load(
            K + ((b * KV_LEN + idx[None, :]) * D + offs_d[:, None]),
            mask=d_mask[:, None] & valid[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q[None, :], k_t, input_precision="ieee")
        scores = tl.reshape(scores, (BLOCK_N,)) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(valid, tl.exp(scores - m_new_safe), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)

        tile_acc = tl.zeros((BLOCK_V,), tl.float32)
        for l_start in range(0, LATENT_D, BLOCK_L):
            offs_l = l_start + tl.arange(0, BLOCK_L)
            l_mask = offs_l < LATENT_D
            latent_v = tl.load(
                V + ((b * KV_LEN + idx[:, None]) * D + offs_l[None, :]),
                mask=valid[:, None] & l_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            w_uv = tl.load(
                W_UV + h * WUV_STRIDE_H + offs_l[:, None] * WUV_STRIDE_L + offs_v[None, :] * WUV_STRIDE_V,
                mask=l_mask[:, None] & v_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            projected_v = tl.dot(latent_v, w_uv, input_precision="ieee")
            tile_acc += tl.sum(p[:, None] * projected_v, axis=0)

        acc = acc * alpha + tile_acc
        m_i = m_new
        l_i = l_new

    part_base = (((b * Q_LEN + q_pos) * H + h) * v_blocks + v_block) * SPLIT_N + split_id
    tl.store(PART_M + part_base, m_i)
    tl.store(PART_L + part_base, l_i)
    tl.store(PART_ACC + part_base * BLOCK_V + tl.arange(0, BLOCK_V), acc, mask=v_mask)


@triton.jit
def _sparse_mla_projected_split_reduce_kernel(
    PART_M,
    PART_L,
    PART_ACC,
    OUT,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    H: tl.constexpr,
    VALUE_D: tl.constexpr,
    SPLIT_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    v_blocks: tl.constexpr = tl.cdiv(VALUE_D, BLOCK_V)
    v_block = pid % v_blocks
    h = (pid // v_blocks) % H
    q_pos = (pid // (v_blocks * H)) % Q_LEN
    b = pid // (v_blocks * H * Q_LEN)

    offs_s = tl.arange(0, BLOCK_S)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    s_mask = offs_s < SPLIT_N
    v_mask = offs_v < VALUE_D

    base = (((b * Q_LEN + q_pos) * H + h) * v_blocks + v_block) * SPLIT_N
    m_s = tl.load(PART_M + base + offs_s, mask=s_mask, other=-float("inf"))
    l_s = tl.load(PART_L + base + offs_s, mask=s_mask, other=0.0)
    m = tl.max(m_s, axis=0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)
    weights = tl.where((l_s > 0.0) & s_mask, tl.exp(m_s - m_safe), 0.0)
    l = tl.sum(weights * l_s, axis=0)
    acc_s = tl.load(
        PART_ACC + (base + offs_s[:, None]) * BLOCK_V + tl.arange(0, BLOCK_V)[None, :],
        mask=s_mask[:, None] & v_mask[None, :],
        other=0.0,
    )
    acc = tl.sum(weights[:, None] * acc_s, axis=0)
    out = acc / tl.where(l == 0.0, 1.0, l)
    tl.store(OUT + (((b * Q_LEN + q_pos) * H + h) * VALUE_D + offs_v), out, mask=v_mask)


@triton.jit
def _sparse_mla_projected_backward_kernel(
    Q,
    K,
    V,
    W_UV,
    TOPK,
    VALID,
    OUT,
    DOUT,
    DQ,
    DK,
    DV,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LATENT_D: tl.constexpr,
    VALUE_D: tl.constexpr,
    WUV_STRIDE_H: tl.constexpr,
    WUV_STRIDE_V: tl.constexpr,
    WUV_STRIDE_L: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    q_pos = (pid // H) % Q_LEN
    b = pid // (H * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    offs_v_base = tl.arange(0, BLOCK_V)
    d_mask = offs_d < D
    latent_mask = offs_d < LATENT_D

    q = tl.load(Q + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0)

    delta = tl.full((), 0.0, tl.float32)
    dlatent = tl.zeros((BLOCK_D,), tl.float32)
    for v_start in range(0, VALUE_D, BLOCK_V):
        offs_v = v_start + offs_v_base
        v_mask = offs_v < VALUE_D
        dout_v = tl.load(
            DOUT + (((b * Q_LEN + q_pos) * H + h) * VALUE_D + offs_v),
            mask=v_mask,
            other=0.0,
        )
        out_v = tl.load(
            OUT + (((b * Q_LEN + q_pos) * H + h) * VALUE_D + offs_v),
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)
        delta += tl.sum(out_v * dout_v.to(tl.float32), axis=0)
        w_uv = tl.load(
            W_UV + h * WUV_STRIDE_H + offs_v[:, None] * WUV_STRIDE_V + offs_d[None, :] * WUV_STRIDE_L,
            mask=v_mask[:, None] & latent_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        dlatent += tl.reshape(tl.dot(dout_v[None, :], w_uv, input_precision="ieee"), (BLOCK_D,))
    dlatent = tl.where(latent_mask, dlatent, 0.0)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        valid = tl.load(
            VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int1)
        k_t = tl.load(
            K + ((b * KV_LEN + idx[None, :]) * D + offs_d[:, None]),
            mask=d_mask[:, None] & valid[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q[None, :], k_t, input_precision="ieee")
        scores = tl.reshape(scores, (BLOCK_N,)) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(valid, tl.exp(scores - m_new_safe), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    dq = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        valid = tl.load(
            VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int1)
        k_ptrs = K + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :])
        v_ptrs = V + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :])
        k = tl.load(
            k_ptrs,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        v = tl.load(
            v_ptrs,
            mask=valid[:, None] & latent_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q[None, :], tl.trans(k), input_precision="ieee")
        scores = tl.reshape(scores, (BLOCK_N,)) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        p = tl.where(valid, tl.exp(scores - m_i_safe), 0.0) / tl.where(l_i == 0.0, 1.0, l_i)
        dp = tl.dot(v.to(tl.float32), dlatent[:, None], input_precision="ieee")
        dp = tl.reshape(dp, (BLOCK_N,))
        ds = p * (dp - delta)
        ds_scaled = ds * SCALE

        dq += tl.reshape(tl.dot(ds_scaled[None, :], k.to(tl.float32), input_precision="ieee"), (BLOCK_D,))
        dk = ds_scaled[:, None] * q[None, :]
        dv = p[:, None] * dlatent[None, :]

        tl.atomic_add(
            DK + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            dk,
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            DV + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            dv,
            mask=valid[:, None] & latent_mask[None, :],
        )

    tl.store(DQ + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), dq, mask=d_mask)


@triton.jit
def _sparse_mla_forward_kernel(
    Q,
    K,
    V,
    TOPK,
    VALID,
    OUT,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_ALL: tl.constexpr,
):
    pid = tl.program_id(0)
    h_block = pid % tl.cdiv(H, BLOCK_H)
    q_pos = (pid // tl.cdiv(H, BLOCK_H)) % Q_LEN
    b = pid // (tl.cdiv(H, BLOCK_H) * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    d_mask = offs_d < D
    h_mask = offs_h < H

    q = tl.load(
        Q + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_H,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        if VALID_ALL:
            valid = n_mask
        else:
            valid = tl.load(
                VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.int1)

        k_t = tl.load(
            K + ((b * KV_LEN + idx[None, :]) * D + offs_d[:, None]),
            mask=d_mask[:, None] & valid[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q, k_t, input_precision="ieee") * SCALE
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(h_mask[:, None] & valid[None, :], tl.exp(scores - m_new_safe[:, None]), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision="ieee")
        m_i = m_new
        l_i = l_new

    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(
        OUT + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _sparse_mla_backward_kernel(
    Q,
    K,
    V,
    TOPK,
    VALID,
    OUT,
    DOUT,
    DQ,
    DK,
    DV,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_ALL: tl.constexpr,
):
    pid = tl.program_id(0)
    h_block = pid % tl.cdiv(H, BLOCK_H)
    q_pos = (pid // tl.cdiv(H, BLOCK_H)) % Q_LEN
    b = pid // (tl.cdiv(H, BLOCK_H) * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    d_mask = offs_d < D
    h_mask = offs_h < H

    q = tl.load(
        Q + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    out = tl.load(
        OUT + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOUT + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    delta = tl.sum(out * dout.to(tl.float32), axis=1)

    m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_H,), 0.0, tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        if VALID_ALL:
            valid = n_mask
        else:
            valid = tl.load(
                VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.int1)
        k_t = tl.load(
            K + ((b * KV_LEN + idx[None, :]) * D + offs_d[:, None]),
            mask=d_mask[:, None] & valid[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q, k_t, input_precision="ieee") * SCALE
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(h_mask[:, None] & valid[None, :], tl.exp(scores - m_new_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    dq = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        if VALID_ALL:
            valid = n_mask
        else:
            valid = tl.load(
                VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.int1)
        k_ptrs = K + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :])
        v_ptrs = V + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :])
        k = tl.load(
            k_ptrs,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        v = tl.load(
            v_ptrs,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, -float("inf"))
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        p = tl.where(h_mask[:, None] & valid[None, :], tl.exp(scores - m_i_safe[:, None]), 0.0) / tl.where(
            l_i == 0.0, 1.0, l_i
        )[:, None]
        dp = tl.dot(dout.to(tl.float32), tl.trans(v.to(tl.float32)), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        ds_scaled = ds * SCALE

        dq += tl.dot(ds_scaled, k.to(tl.float32), input_precision="ieee")
        # Aggregate all heads in this tile before the HBM atomic, reducing atomic traffic by BLOCK_H.
        dk = tl.dot(tl.trans(ds_scaled), q.to(tl.float32), input_precision="ieee")
        dv = tl.dot(tl.trans(p), dout.to(tl.float32), input_precision="ieee")

        tl.atomic_add(
            DK + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            dk,
            mask=valid[:, None] & d_mask[None, :],
        )
        tl.atomic_add(
            DV + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            dv,
            mask=valid[:, None] & d_mask[None, :],
        )

    tl.store(
        DQ + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
        dq,
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _sparse_mla_forward_persistent_kernel(
    Q,
    K,
    V,
    TOPK,
    VALID,
    OUT,
    B: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    TOTAL_WORK: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    work_id = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    head_blocks: tl.constexpr = tl.cdiv(H, BLOCK_H)

    while work_id < TOTAL_WORK:
        h_block = work_id % head_blocks
        q_pos = (work_id // head_blocks) % Q_LEN
        b = work_id // (head_blocks * Q_LEN)
        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H

        q = tl.load(
            Q + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.full((BLOCK_H,), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        for start in range(0, TOPK_N, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < TOPK_N
            idx = tl.load(
                TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            )
            valid = tl.load(
                VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.int1)

            k_t = tl.load(
                K + ((b * KV_LEN + idx[None, :]) * D + offs_d[:, None]),
                mask=d_mask[:, None] & valid[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            scores = tl.dot(q, k_t, input_precision="ieee") * SCALE
            scores = tl.where(h_mask[:, None] & valid[None, :], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
            p = tl.where(h_mask[:, None] & valid[None, :], tl.exp(scores - m_new_safe[:, None]), 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                V + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32), input_precision="ieee")
            m_i = m_new
            l_i = l_new

        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(
            OUT + (((b * Q_LEN + q_pos) * H + offs_h[:, None]) * D + offs_d[None, :]),
            out,
            mask=h_mask[:, None] & d_mask[None, :],
        )
        work_id += NUM_PROGRAMS


class _SparseMLATritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, valid, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        topk = topk.contiguous()
        valid = valid.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        block_d, block_n, block_h, num_warps, num_stages = _get_sparse_mla_kernel_config(head_dim, topk_n)
        out = torch.empty_like(q)
        head_blocks = triton.cdiv(num_heads, block_h)
        _sparse_mla_forward_kernel[(batch_size * query_len * head_blocks,)](
            q,
            k,
            v,
            topk,
            valid,
            out,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            scale,
            block_n,
            block_d,
            block_h,
            False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, topk, valid, out)
        ctx.scale = scale
        ctx.block_d = block_d
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, topk, valid, out = ctx.saved_tensors
        dout = dout.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        head_blocks = triton.cdiv(num_heads, ctx.block_h)
        _sparse_mla_backward_kernel[(batch_size * query_len * head_blocks,)](
            q,
            k,
            v,
            topk,
            valid,
            out,
            dout,
            dq,
            dk,
            dv,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            ctx.scale,
            ctx.block_n,
            ctx.block_d,
            ctx.block_h,
            False,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def sparse_mla_triton(q, k, v, topk, valid, scale):
    return _SparseMLATritonFunction.apply(q, k, v, topk, valid, scale)


class _SparseMLAValidAllTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, valid, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        topk = topk.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        block_d, block_n, block_h, num_warps, num_stages = _get_sparse_mla_kernel_config(head_dim, topk_n)
        out = torch.empty_like(q)
        head_blocks = triton.cdiv(num_heads, block_h)
        _sparse_mla_forward_kernel[(batch_size * query_len * head_blocks,)](
            q,
            k,
            v,
            topk,
            valid,
            out,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            scale,
            block_n,
            block_d,
            block_h,
            True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, topk, out)
        ctx.scale = scale
        ctx.block_d = block_d
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, topk, out = ctx.saved_tensors
        dout = dout.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        head_blocks = triton.cdiv(num_heads, ctx.block_h)
        _sparse_mla_backward_kernel[(batch_size * query_len * head_blocks,)](
            q,
            k,
            v,
            topk,
            topk,
            out,
            dout,
            dq,
            dk,
            dv,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            ctx.scale,
            ctx.block_n,
            ctx.block_d,
            ctx.block_h,
            True,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def sparse_mla_triton_valid_all(q, k, v, topk, valid, scale):
    return _SparseMLAValidAllTritonFunction.apply(q, k, v, topk, valid, scale)


class _SparseMLAPersistentTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, valid, scale, num_programs):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        topk = topk.contiguous()
        valid = valid.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        block_d, block_n, block_h, num_warps, num_stages = _get_sparse_mla_kernel_config(head_dim, topk_n)
        head_blocks = triton.cdiv(num_heads, block_h)
        total_work = batch_size * query_len * head_blocks
        out = torch.empty_like(q)
        _sparse_mla_forward_persistent_kernel[(num_programs,)](
            q,
            k,
            v,
            topk,
            valid,
            out,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            scale,
            total_work,
            num_programs,
            block_n,
            block_d,
            block_h,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, topk, valid, out)
        ctx.scale = scale
        ctx.block_d = block_d
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, topk, valid, out = ctx.saved_tensors
        dout = dout.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        head_blocks = triton.cdiv(num_heads, ctx.block_h)
        _sparse_mla_backward_kernel[(batch_size * query_len * head_blocks,)](
            q,
            k,
            v,
            topk,
            valid,
            out,
            dout,
            dq,
            dk,
            dv,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            ctx.scale,
            ctx.block_n,
            ctx.block_d,
            ctx.block_h,
            False,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def sparse_mla_triton_persistent(q, k, v, topk, valid, scale, num_programs=None):
    if num_programs is None:
        props = torch.cuda.get_device_properties(q.device)
        num_programs = props.multi_processor_count
    return _SparseMLAPersistentTritonFunction.apply(q, k, v, topk, valid, scale, num_programs)


class _SparseMLABF16KVGradTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, valid, scale):
        out = sparse_mla_triton(q, k, v, topk, valid, scale)
        ctx.save_for_backward(q.contiguous(), k.contiguous(), v.contiguous(), topk.contiguous(), valid.contiguous(), out)
        ctx.scale = scale
        batch_size, query_len, num_heads, head_dim = q.shape
        topk_n = topk.shape[-1]
        block_d, block_n, block_h, num_warps, num_stages = _get_sparse_mla_kernel_config(head_dim, topk_n)
        ctx.block_d = block_d
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, topk, valid, out = ctx.saved_tensors
        dout = dout.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        dq = torch.empty_like(q, dtype=torch.float32)
        # Experimental: use bf16 K/V gradient buffers so atomics write half the bytes.
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        head_blocks = triton.cdiv(num_heads, ctx.block_h)
        _sparse_mla_backward_kernel[(batch_size * query_len * head_blocks,)](
            q,
            k,
            v,
            topk,
            valid,
            out,
            dout,
            dq,
            dk,
            dv,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            topk_n,
            ctx.scale,
            ctx.block_n,
            ctx.block_d,
            ctx.block_h,
            False,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        return dq.to(q.dtype), dk, dv, None, None, None


def sparse_mla_triton_bf16_kv_grad(q, k, v, topk, valid, scale):
    return _SparseMLABF16KVGradTritonFunction.apply(q, k, v, topk, valid, scale)


class _SparseMLAProjectedTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk, valid, w_uv, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        topk = topk.contiguous()
        valid = valid.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        latent_dim = w_uv.shape[-1]
        value_dim = w_uv.shape[-2]
        block_d, block_n, block_l, block_v, num_warps, num_stages = _get_sparse_mla_projected_kernel_config(
            head_dim, value_dim, topk_n
        )
        out = torch.empty(
            batch_size,
            query_len,
            num_heads,
            value_dim,
            device=q.device,
            dtype=q.dtype,
        )
        _sparse_mla_projected_forward_kernel[
            (batch_size * query_len * num_heads * triton.cdiv(value_dim, block_v),)
        ](
            q,
            k,
            v,
            w_uv,
            topk,
            valid,
            out,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            latent_dim,
            value_dim,
            w_uv.stride(0),
            w_uv.stride(1),
            w_uv.stride(2),
            topk_n,
            scale,
            block_n,
            block_d,
            block_l,
            block_v,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, topk, valid, w_uv, out)
        ctx.scale = scale
        ctx.block_d = block_d
        ctx.block_n = block_n
        ctx.block_v = block_v
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, topk, valid, w_uv, out = ctx.saved_tensors
        dout = dout.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]
        topk_n = topk.shape[-1]
        latent_dim = w_uv.shape[-1]
        value_dim = w_uv.shape[-2]
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        with torch.no_grad():
            latent = sparse_mla_triton(q, k, v, topk, valid, ctx.scale)[..., :latent_dim]
            dw_uv = torch.einsum("bshv,bshl->hvl", dout, latent)
        _sparse_mla_projected_backward_kernel[(batch_size * query_len * num_heads,)](
            q,
            k,
            v,
            w_uv,
            topk,
            valid,
            out,
            dout,
            dq,
            dk,
            dv,
            batch_size,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            latent_dim,
            value_dim,
            w_uv.stride(0),
            w_uv.stride(1),
            w_uv.stride(2),
            topk_n,
            ctx.scale,
            ctx.block_n,
            ctx.block_d,
            ctx.block_v,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, dw_uv.to(w_uv.dtype), None


def sparse_mla_triton_projected(q, k, v, topk, valid, w_uv, scale):
    return _SparseMLAProjectedTritonFunction.apply(q, k, v, topk, valid, w_uv, scale)


# =============================================================================
# Latent-space sparse MLA (v2)
#
# Key differences from the kernels above:
#   - V-as-K fusion: DeepSeek MLA's value rows are the first LATENT_D columns
#     of the key rows (the model builds value_states = pad(kv_c_normed) and
#     key_states = cat(kv_c_normed, k_rot)). The kernels below gather each key
#     row once and reuse its latent slice as V, halving gather traffic, and the
#     K/V gradients collapse into ONE atomic buffer instead of two.
#   - Latent output: returns (B, Q, H, LATENT_D) directly (callers always slice
#     [..., :kv_lora_rank] anyway), so no compute is spent on the zero rope
#     columns of the padded V.
#   - Tensor-core dots: all GEMMs run on the native (bf16/fp16) operands with
#     fp32 accumulators — flash-attention-2 semantics — instead of the fp32
#     `input_precision="ieee"` CUDA-core dots of the v1 backward (~20x below
#     the bf16 tensor-core rate on H200/B300).
#   - Saved LSE: forward stores per-(b, q, h) logsumexp; backward reconstructs
#     softmax probabilities from it instead of re-running the online-softmax
#     pass over all gathered keys (removes one of the v1 backward's two
#     score passes).
#   - Larger head tiles: MLA shares each key row across all H=128 query heads,
#     so BLOCK_H amortizes the random-index gathers; the padded-BLOCK_D dot of
#     v1 (1024-wide for a 576-dim head) is replaced by exact-width dots on the
#     512/64 latent/rope split.
#
# sm_103 (B300) notes — see triton-lang/triton#10821:
#   Two tl.dot calls chained through ONE accumulator in a K-loop miscompile on
#   the tcgen05 MMA path (BLOCK_M >= 64). Every loop below issues exactly one
#   tl.dot per accumulator per iteration; the latent+rope score split uses two
#   SEPARATE accumulators that are summed afterwards (the verified-correct
#   control in the bug report), never `acc = dot(...); acc = dot(..., acc)`.
# =============================================================================


@triton.jit
def _sparse_mla_latent_fwd_kernel(
    Q,
    K,
    TOPK,
    VALID,
    OUT,
    LSE,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LATENT_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_ALL: tl.constexpr,
):
    pid = tl.program_id(0)
    head_blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    h_block = pid % head_blocks
    q_pos = (pid // head_blocks) % Q_LEN
    b = pid // (head_blocks * Q_LEN)

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_lat = tl.arange(0, LATENT_D)
    offs_rope = tl.arange(0, ROPE_D)
    h_mask = offs_h < H
    q_row0 = (b * Q_LEN + q_pos) * H

    # Whole q tile for this head block, split into latent / rope halves.
    q_lat = tl.load(
        Q + (q_row0 + offs_h[:, None]) * D + offs_lat[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q_rope = tl.load(
        Q + (q_row0 + offs_h[:, None]) * D + (LATENT_D + offs_rope[None, :]),
        mask=h_mask[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, LATENT_D), tl.float32)

    for start in tl.range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        if VALID_ALL:
            valid = n_mask
        else:
            valid = tl.load(
                VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.int1)
        k_row = (b * KV_LEN + idx).to(tl.int64)

        # One natural-layout gather per key row; the latent slice doubles as V.
        k_lat = tl.load(
            K + k_row[:, None] * D + offs_lat[None, :],
            mask=valid[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        k_rope = tl.load(
            K + k_row[:, None] * D + (LATENT_D + offs_rope[None, :]),
            mask=valid[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        # Two separate accumulators summed — NOT one accumulator chained
        # through two dots (sm_103 tcgen05 miscompile, triton#10821).
        s_lat = tl.dot(q_lat, tl.trans(k_lat))
        s_rope = tl.dot(q_rope, tl.trans(k_rope))
        scores = (s_lat + s_rope) * SCALE
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(h_mask[:, None] & valid[None, :], tl.exp(scores - m_new_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(k_lat.dtype), k_lat, acc)  # single dot per acc per iteration
        m_i = m_new

    # lse = +inf for fully-masked rows so backward reconstructs p = 0 there.
    m_final_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
    lse = tl.where(l_i > 0.0, m_final_safe + tl.log(l_i), float("inf"))
    tl.store(LSE + (b * Q_LEN + q_pos) * H + offs_h, lse, mask=h_mask)

    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(
        OUT + (q_row0 + offs_h[:, None]) * LATENT_D + offs_lat[None, :],
        out.to(OUT.dtype.element_ty),
        mask=h_mask[:, None],
    )


@triton.jit
def _sparse_mla_latent_bwd_kernel(
    Q,
    K,
    TOPK,
    VALID,
    DOUT,
    LSE,
    DELTA,
    DQ,
    DKV,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    LATENT_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    TOPK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VALID_ALL: tl.constexpr,
):
    pid = tl.program_id(0)
    head_blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    h_block = pid % head_blocks
    q_pos = (pid // head_blocks) % Q_LEN
    b = pid // (head_blocks * Q_LEN)

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_lat = tl.arange(0, LATENT_D)
    offs_rope = tl.arange(0, ROPE_D)
    h_mask = offs_h < H
    q_row0 = (b * Q_LEN + q_pos) * H

    q_lat = tl.load(
        Q + (q_row0 + offs_h[:, None]) * D + offs_lat[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q_rope = tl.load(
        Q + (q_row0 + offs_h[:, None]) * D + (LATENT_D + offs_rope[None, :]),
        mask=h_mask[:, None],
        other=0.0,
    )
    dout = tl.load(
        DOUT + (q_row0 + offs_h[:, None]) * LATENT_D + offs_lat[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    # lse = +inf (=> p = 0) for fully-masked and out-of-range head rows.
    lse = tl.load(LSE + (b * Q_LEN + q_pos) * H + offs_h, mask=h_mask, other=float("inf"))
    delta = tl.load(DELTA + (b * Q_LEN + q_pos) * H + offs_h, mask=h_mask, other=0.0)

    dq_lat = tl.zeros((BLOCK_H, LATENT_D), tl.float32)
    dq_rope = tl.zeros((BLOCK_H, ROPE_D), tl.float32)

    for start in tl.range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(
            TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
            mask=n_mask,
            other=0,
            eviction_policy="evict_first",
        )
        if VALID_ALL:
            valid = n_mask
        else:
            valid = tl.load(
                VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n,
                mask=n_mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.int1)
        k_row = (b * KV_LEN + idx).to(tl.int64)

        k_lat = tl.load(
            K + k_row[:, None] * D + offs_lat[None, :],
            mask=valid[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        k_rope = tl.load(
            K + k_row[:, None] * D + (LATENT_D + offs_rope[None, :]),
            mask=valid[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        # Separate accumulators summed (sm_103-safe; see module notes).
        s_lat = tl.dot(q_lat, tl.trans(k_lat))
        s_rope = tl.dot(q_rope, tl.trans(k_rope))
        scores = (s_lat + s_rope) * SCALE
        # p from saved lse — no online-softmax recompute pass.
        p = tl.where(
            h_mask[:, None] & valid[None, :],
            tl.exp(scores - lse[:, None]),
            0.0,
        )

        dp = tl.dot(dout, tl.trans(k_lat))  # fresh accumulator
        ds = p * (dp - delta[:, None]) * SCALE

        p16 = p.to(k_lat.dtype)
        ds16 = ds.to(k_lat.dtype)

        dq_lat = tl.dot(ds16, k_lat, dq_lat)  # single dot per acc per iteration
        dq_rope = tl.dot(ds16, k_rope, dq_rope)  # single dot per acc per iteration

        # dK (latent) + dV share the key row: two fresh accumulators, summed
        # outside the dots, one fp32 atomic per element.
        dk_lat = tl.dot(tl.trans(ds16), q_lat)
        dv_lat = tl.dot(tl.trans(p16), dout)
        tl.atomic_add(
            DKV + k_row[:, None] * D + offs_lat[None, :],
            dk_lat + dv_lat,
            mask=valid[:, None],
        )
        dk_rope = tl.dot(tl.trans(ds16), q_rope)
        tl.atomic_add(
            DKV + k_row[:, None] * D + (LATENT_D + offs_rope[None, :]),
            dk_rope,
            mask=valid[:, None],
        )

    tl.store(
        DQ + (q_row0 + offs_h[:, None]) * D + offs_lat[None, :],
        dq_lat.to(DQ.dtype.element_ty),
        mask=h_mask[:, None],
    )
    tl.store(
        DQ + (q_row0 + offs_h[:, None]) * D + (LATENT_D + offs_rope[None, :]),
        dq_rope.to(DQ.dtype.element_ty),
        mask=h_mask[:, None],
    )


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


@functools.cache
def _latent_mla_configs(valid_all: bool = True):
    """(BLOCK_H, BLOCK_N, num_warps, num_stages) for (forward, backward).

    Measured at DeepSeek-3.2 shapes (H=128, D=576, topk=2048): sm_90 on H200,
    sm_100/sm_103 on B300 (subprocess-isolated correctness + timing sweep over
    BLOCK_H/BLOCK_N/warps/stages). DSMLA_* env vars override for tuning
    experiments.

    sm_103 (B300, Triton 3.6) hard constraints found by the sweep — these are
    launch/compile failures, not tuning preferences:
      - forward BLOCK_H=64: every variant either miscompiles into misaligned
        tcgen05 addresses (BN<=32) or needs 352-632 KB SMEM vs the 227 KB
        budget (BN>=64), so Blackwell forward tiles cap at BLOCK_H=32.
        BLOCK_H=128 exceeds the 512-column TMEM budget outright.
      - backward BLOCK_N>=64: the extra dp/dk/dv accumulators exceed TMEM
        (576 cols needed vs 512), so backward keeps BLOCK_N<=32.
    """
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if capability[0] == 9:
        # H200-measured optimum (fwd 0.37 ms / bwd 4.2 ms per 128-query chunk
        # at topk=2048).
        fwd = (64, 64, 8, 2)
        bwd = (32, 32, 8, 2)
    elif capability[0] == 10:
        # B300-measured (sm_103). Valid-all chunks (steady state for long
        # sequences) run fastest at BN=64/S=3 (0.28 ms fwd, 4.4 ms bwd);
        # mixed-validity chunks (early positions with masked lanes) prefer
        # BN=128/W=8 forward (0.52 vs 0.79 ms) and the small BH=BN=16 tile
        # backward (2.12 vs 2.32 ms).
        fwd = (32, 64, 4, 3) if valid_all else (32, 128, 8, 2)
        bwd = (32, 32, 8, 2) if valid_all else (16, 16, 4, 2)
    else:
        fwd = (16, 32, 4, 2)
        bwd = (16, 32, 4, 2)
    fwd = (
        _env_int("DSMLA_FWD_BH", fwd[0]),
        _env_int("DSMLA_FWD_BN", fwd[1]),
        _env_int("DSMLA_FWD_WARPS", fwd[2]),
        _env_int("DSMLA_FWD_STAGES", fwd[3]),
    )
    bwd = (
        _env_int("DSMLA_BWD_BH", bwd[0]),
        _env_int("DSMLA_BWD_BN", bwd[1]),
        _env_int("DSMLA_BWD_WARPS", bwd[2]),
        _env_int("DSMLA_BWD_STAGES", bwd[3]),
    )
    return fwd, bwd


def _latent_kernel_supported(latent_d: int, rope_d: int) -> bool:
    """The v2 kernels need power-of-2 latent/rope widths that satisfy tl.dot's
    minimum tile dims. Real DeepSeek-3.2 (512/64) qualifies; tiny test configs
    fall back to the v1 kernels."""
    def _pow2(n):
        return n >= 16 and (n & (n - 1)) == 0

    return _pow2(latent_d) and _pow2(rope_d)


class _SparseMLALatentTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, topk, valid, scale, latent_d):
        q = q.contiguous()
        kv = kv.contiguous()
        topk = topk.contiguous()
        valid_all = valid is None
        if not valid_all:
            valid = valid.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = kv.shape[1]
        topk_n = topk.shape[-1]
        rope_d = head_dim - latent_d

        (block_h, block_n, num_warps, num_stages), _ = _latent_mla_configs(valid_all)
        out = torch.empty(batch_size, query_len, num_heads, latent_d, device=q.device, dtype=q.dtype)
        lse = torch.empty(batch_size, query_len, num_heads, device=q.device, dtype=torch.float32)
        head_blocks = triton.cdiv(num_heads, block_h)
        _sparse_mla_latent_fwd_kernel[(batch_size * query_len * head_blocks,)](
            q,
            kv,
            topk,
            topk if valid_all else valid,
            out,
            lse,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            latent_d,
            rope_d,
            topk_n,
            scale,
            block_n,
            block_h,
            valid_all,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, kv, topk, out, lse)
        ctx.valid = valid if not valid_all else None
        ctx.scale = scale
        ctx.latent_d = latent_d
        return out

    @staticmethod
    def backward(ctx, dout):
        q, kv, topk, out, lse = ctx.saved_tensors
        valid = ctx.valid
        valid_all = valid is None
        dout = dout.contiguous()
        batch_size, query_len, num_heads, head_dim = q.shape
        kv_len = kv.shape[1]
        topk_n = topk.shape[-1]
        latent_d = ctx.latent_d
        rope_d = head_dim - latent_d

        # delta = rowsum(out * dout) per (b, q, h) — one cheap fused pass.
        delta = (out.float() * dout.float()).sum(dim=-1)

        _, (block_h, block_n, num_warps, num_stages) = _latent_mla_configs(valid_all)
        dq = torch.empty_like(q)
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        head_blocks = triton.cdiv(num_heads, block_h)
        _sparse_mla_latent_bwd_kernel[(batch_size * query_len * head_blocks,)](
            q,
            kv,
            topk,
            topk if valid_all else valid,
            dout,
            lse,
            delta,
            dq,
            dkv,
            query_len,
            kv_len,
            num_heads,
            head_dim,
            latent_d,
            rope_d,
            topk_n,
            ctx.scale,
            block_n,
            block_h,
            valid_all,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return dq, dkv.to(kv.dtype), None, None, None, None


def sparse_mla_latent_triton(q, kv, topk, valid, scale, latent_d=512):
    """Sparse MLA over shared latent K/V rows, returning the latent output.

    Args:
        q:     (B, Q, H, D) queries, D = latent_d + rope_d.
        kv:    (B, KV, D) key rows = cat(latent, rope); the latent slice
               doubles as the value matrix (no separate V input).
        topk:  (B, Q, topk_n) int32 sorted key indices.
        valid: (B, Q, topk_n) bool validity mask, or None if every index is
               valid (fast path, skips the mask loads).
        scale: softmax scale.
        latent_d: width of the latent slice (DeepSeek-3.2: kv_lora_rank=512).

    Returns:
        (B, Q, H, latent_d) attention output in q.dtype. Gradients flow to q
        and kv (the K and V roles of the latent slice are fused into one
        gradient buffer).
    """
    rope_d = q.shape[-1] - latent_d
    if not _latent_kernel_supported(latent_d, rope_d):
        # Fallback for tiny/non-power-of-2 test configs: v1 kernels on a
        # zero-padded V (autograd routes the pad/slice gradients).
        value = torch.nn.functional.pad(kv[..., :latent_d], (0, rope_d))
        if valid is None:
            return _SparseMLAValidAllTritonFunction.apply(q, kv, value, topk, topk, scale)[..., :latent_d]
        return _SparseMLATritonFunction.apply(q, kv, value, topk, valid, scale)[..., :latent_d]
    return _SparseMLALatentTritonFunction.apply(q, kv, topk, valid, scale, latent_d)


def sparse_mla_triton_projected_split(q, k, v, topk, valid, w_uv, scale, split_n=8):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    topk = topk.contiguous()
    valid = valid.contiguous()
    batch_size, query_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]
    topk_n = topk.shape[-1]
    latent_dim = w_uv.shape[-1]
    value_dim = w_uv.shape[-2]
    block_d, block_n, block_l, block_v, num_warps, num_stages = _get_sparse_mla_projected_kernel_config(
        head_dim, value_dim, topk_n
    )
    split_n = min(split_n, triton.next_power_of_2(topk_n))
    split_size = triton.cdiv(topk_n, split_n)
    split_size = triton.cdiv(split_size, block_n) * block_n
    value_blocks = triton.cdiv(value_dim, block_v)
    part_shape = (batch_size, query_len, num_heads, value_blocks, split_n)
    part_m = torch.empty(part_shape, device=q.device, dtype=torch.float32)
    part_l = torch.empty(part_shape, device=q.device, dtype=torch.float32)
    part_acc = torch.empty((*part_shape, block_v), device=q.device, dtype=torch.float32)
    out = torch.empty(batch_size, query_len, num_heads, value_dim, device=q.device, dtype=q.dtype)
    _sparse_mla_projected_split_stage1_kernel[(batch_size * query_len * num_heads * value_blocks * split_n,)](
        q,
        k,
        v,
        w_uv,
        topk,
        valid,
        part_m,
        part_l,
        part_acc,
        batch_size,
        query_len,
        kv_len,
        num_heads,
        head_dim,
        latent_dim,
        value_dim,
        w_uv.stride(0),
        w_uv.stride(1),
        w_uv.stride(2),
        topk_n,
        scale,
        split_n,
        split_size,
        block_n,
        block_d,
        block_l,
        block_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _sparse_mla_projected_split_reduce_kernel[(batch_size * query_len * num_heads * value_blocks,)](
        part_m,
        part_l,
        part_acc,
        out,
        batch_size,
        query_len,
        num_heads,
        value_dim,
        split_n,
        triton.next_power_of_2(split_n),
        block_v,
        num_warps=1,
    )
    return out
