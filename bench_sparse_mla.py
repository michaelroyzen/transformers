"""Accuracy + performance harness for the DeepSeek-V3.2 sparse MLA Triton kernels.

Compares kernels against an fp32 eager reference at real DeepSeek-3.2 shapes
(H=128 heads, D=576 = 512 latent + 64 rope, topk=2048), for both the
valid-all fast path and the mixed-validity path.

Usage:
    LD_LIBRARY_PATH= PYTHONPATH=src .venv/bin/python bench_sparse_mla.py [--kv 16384]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from transformers.models.deepseek_v32.sparse_mla_triton import (  # noqa: E402
    sparse_mla_triton,
    sparse_mla_triton_valid_all,
)

LATENT_D = 512
ROPE_D = 64
D = LATENT_D + ROPE_D


def make_inputs(batch, q_len, kv_len, heads, topk, device, dtype, valid_all):
    torch.manual_seed(0)
    q = torch.randn(batch, q_len, heads, D, device=device, dtype=dtype) / 3
    kv = torch.randn(batch, kv_len, D, device=device, dtype=dtype) / 3
    # value = latent part of key, zero-padded to D (what the model does)
    v = torch.nn.functional.pad(kv[..., :LATENT_D], (0, ROPE_D))

    if valid_all:
        # all q positions attend to `topk` fully-valid keys (post-warmup chunks)
        q_positions = torch.arange(kv_len - q_len, kv_len, device=device)
    else:
        # early chunk: causal prefix < topk for some rows -> invalid lanes
        q_positions = torch.arange(q_len, device=device)

    scores = torch.randn(batch, q_len, kv_len, device=device)
    key_positions = torch.arange(kv_len, device=device)
    causal = key_positions[None, None, :] <= q_positions[None, :, None]
    scores = scores.masked_fill(~causal, float("-inf"))
    k_eff = min(topk, kv_len)
    topk_indices = scores.topk(k_eff, dim=-1).indices.to(torch.int32)
    topk_indices, _ = topk_indices.sort(dim=-1)
    valid = causal.expand(batch, q_len, kv_len).gather(-1, topk_indices.long())
    return q, kv, v, topk_indices, valid


def eager_reference_fp32(q, kv, topk_indices, valid, scale):
    """fp32 gather + softmax + P@V_latent reference. Returns (out_latent, lse)."""
    B, Q, H, _ = q.shape
    topk = topk_indices.shape[-1]
    idx = topk_indices.long()
    k_sel = kv.float().gather(1, idx.reshape(B, -1)[..., None].expand(-1, -1, D)).reshape(B, Q, topk, D)
    scores = torch.einsum("bqhd,bqtd->bhqt", q.float(), k_sel) * scale
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)  # rows with zero valid keys
    out = torch.einsum("bhqt,bqtd->bqhd", p, k_sel[..., :LATENT_D])
    return out


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def run_case(fn, q, kv, v, topk_indices, valid, scale, dout, needs_full_pad_out):
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    v_in = torch.nn.functional.pad(kv[..., :LATENT_D], (0, ROPE_D))
    out = fn(q, kv, v_in, topk_indices, valid, scale)
    out_latent = out[..., :LATENT_D]
    out_latent.backward(dout)
    return out_latent.detach(), q.grad.detach(), kv.grad.detach()


def bench(fn, q, kv, v, topk_indices, valid, scale, dout, iters=20):
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)

    def step():
        v_in = torch.nn.functional.pad(kv[..., :LATENT_D], (0, ROPE_D))
        out = fn(q, kv, v_in, topk_indices, valid, scale)[..., :LATENT_D]
        out.backward(dout)
        q.grad = None
        kv.grad = None

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    fwd_ms = bwd_ms = 0.0
    for _ in range(iters):
        v_in = torch.nn.functional.pad(kv[..., :LATENT_D], (0, ROPE_D))
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        out = fn(q, kv, v_in, topk_indices, valid, scale)[..., :LATENT_D]
        mid.record()
        out.backward(dout)
        end.record()
        torch.cuda.synchronize()
        fwd_ms += start.elapsed_time(mid)
        bwd_ms += mid.elapsed_time(end)
        q.grad = None
        kv.grad = None
    return fwd_ms / iters, bwd_ms / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", type=int, default=16384)
    parser.add_argument("--q", type=int, default=128)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    scale = (128 + 64) ** -0.5  # qk_head_dim^-0.5 (mscale omitted; constant factor)

    for valid_all in (True, False):
        label = "valid_all" if valid_all else "mixed-validity"
        print(f"\n=== {label}: B={args.batch} Q={args.q} KV={args.kv} H={args.heads} topk={args.topk} ===")
        q, kv, v, topk_indices, valid = make_inputs(
            args.batch, args.q, args.kv, args.heads, args.topk, device, dtype, valid_all
        )
        dout = torch.randn(args.batch, args.q, args.heads, LATENT_D, device=device) / 5

        # fp32 eager reference with autograd
        q32 = q.detach().float().requires_grad_(True)
        kv32 = kv.detach().float().requires_grad_(True)
        ref_out = eager_reference_fp32(q32, kv32, topk_indices, valid, scale)
        ref_out.backward(dout)
        ref = (ref_out.detach(), q32.grad.detach(), kv32.grad.detach())

        fns = {"old": sparse_mla_triton_valid_all if valid_all else sparse_mla_triton}
        try:
            from transformers.models.deepseek_v32.sparse_mla_triton import sparse_mla_latent_triton

            def latent_fn(q, kv, v_unused, topk_indices, valid, scale):
                return torch.nn.functional.pad(
                    sparse_mla_latent_triton(q, kv, topk_indices, None if valid_all else valid, scale),
                    (0, ROPE_D),
                )

            fns["new"] = latent_fn
        except ImportError:
            pass

        for name, fn in fns.items():
            out, dq, dkv = run_case(fn, q, kv, v, topk_indices, valid, scale, dout, True)
            errs = (max_err(out, ref[0]), max_err(dq, ref[1]), max_err(dkv, ref[2]))
            fwd_ms, bwd_ms = bench(fn, q, kv, v, topk_indices, valid, scale, dout, args.iters)
            print(
                f"  {name:4s}  out_err={errs[0]:.4e}  dq_err={errs[1]:.4e}  dkv_err={errs[2]:.4e}"
                f"  |  fwd {fwd_ms:7.3f} ms  bwd {bwd_ms:7.3f} ms  tot {fwd_ms + bwd_ms:7.3f} ms"
            )


if __name__ == "__main__":
    main()
