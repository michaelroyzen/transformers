import torch
import triton
import triton.language as tl


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
):
    pid = tl.program_id(0)
    h = pid % H
    q_pos = (pid // H) % Q_LEN
    b = pid // (H * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q = tl.load(Q + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n, mask=n_mask, other=0).to(tl.int64)
        valid = tl.load(VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n, mask=n_mask, other=0).to(tl.int1)

        k = tl.load(
            K + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        p = tl.where(valid, tl.exp(scores - m_new_safe), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(
            V + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    out = acc / tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(OUT + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), out, mask=d_mask)


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
):
    pid = tl.program_id(0)
    h = pid % H
    q_pos = (pid // H) % Q_LEN
    b = pid // (H * Q_LEN)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q = tl.load(Q + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0).to(tl.float32)
    out = tl.load(OUT + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), mask=d_mask, other=0.0).to(tl.float32)
    delta = tl.sum(out * dout, axis=0)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)

    for start in range(0, TOPK_N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TOPK_N
        idx = tl.load(TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n, mask=n_mask, other=0).to(tl.int64)
        valid = tl.load(VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n, mask=n_mask, other=0).to(tl.int1)
        k = tl.load(
            K + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :]),
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * SCALE
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
        idx = tl.load(TOPK + (b * Q_LEN + q_pos) * TOPK_N + offs_n, mask=n_mask, other=0).to(tl.int64)
        valid = tl.load(VALID + (b * Q_LEN + q_pos) * TOPK_N + offs_n, mask=n_mask, other=0).to(tl.int1)
        k_ptrs = K + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :])
        v_ptrs = V + ((b * KV_LEN + idx[:, None]) * D + offs_d[None, :])
        k = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        p = tl.where(valid, tl.exp(scores - m_i_safe), 0.0) / tl.where(l_i == 0.0, 1.0, l_i)
        dp = tl.sum(v * dout[None, :], axis=1)
        ds = p * (dp - delta)

        dq += tl.sum((ds * SCALE)[:, None] * k, axis=0)
        dk = (ds * SCALE)[:, None] * q[None, :]
        dv = p[:, None] * dout[None, :]

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

    tl.store(DQ + (((b * Q_LEN + q_pos) * H + h) * D + offs_d), dq, mask=d_mask)


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
        block_d = triton.next_power_of_2(head_dim)
        block_n = 16
        out = torch.empty_like(q)
        _sparse_mla_forward_kernel[(batch_size * query_len * num_heads,)](
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
            num_warps=8,
        )
        ctx.save_for_backward(q, k, v, topk, valid, out)
        ctx.scale = scale
        ctx.block_d = block_d
        ctx.block_n = block_n
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
        _sparse_mla_backward_kernel[(batch_size * query_len * num_heads,)](
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
            num_warps=8,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def sparse_mla_triton(q, k, v, topk, valid, scale):
    return _SparseMLATritonFunction.apply(q, k, v, topk, valid, scale)
