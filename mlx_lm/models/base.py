# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


# ---------------------------------------------------------------------------
# Opt-in: query-chunked attention backward (training-time, long causal only).
#
# MLX's fused SDPA has no GPU backward, so a training forward decomposes into
# B x H x L x L score tensors -- the dominant memory term past seq ~4K (e.g.
# 8 full-attention layers in Qwen3.5). When enabled via set_chunked_attention,
# full-sequence causal training attention keeps a chunked flash-forward and
# computes the backward per query-chunk, holding ~one chunk of scores
# (O(Lc x L)) instead of O(L^2). Default off: inference and any short, padded,
# quantized, or sink attention takes the stock fused path unchanged.
# ---------------------------------------------------------------------------

_CHUNKED_ATTN_Q = 0
_CHUNKED_ATTN_MIN_LEN = 1024
_chunked_sdpa = None

# Cross-layer serialization chain: holds the most recent layer's gradient
# accumulators within one backward pass so layer backwards serialize instead of
# overlapping under checkpoint recompute. Stale entries only add a no-op graph
# edge, so no reset is needed between loss calls.
_chunked_chain: Optional[list] = None


def make_chunked_sdpa(q_chunk: int = 512):
    """Build an SDPA with a chunked flash-forward and chunked analytic backward.

    Supports mask in {"causal", None}; other masks fall back to the fused path.
    """
    if q_chunk <= 0:
        raise ValueError("q_chunk must be positive")

    def sdpa(q, k, v, *, scale, mask):
        if mask is not None and mask != "causal":
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        causal = mask == "causal"

        @mx.custom_function
        def flash(qq, kk, vv):
            # The fused SDPA kernel is inference-only (use_fallback is true under
            # is_training), so a training forward would materialize B x H x L^2
            # scores. Chunked flash-forward keeps the transient at O(Lc x L).
            B, Hq, L, Dh = qq.shape
            Hk = kk.shape[1]
            rep = Hq // Hk
            qf = qq.reshape(B, Hk, rep, L, Dh)
            kf = kk[:, :, None]
            vf = vv[:, :, None]
            outs = []
            for s in range(0, L, q_chunk):
                e = min(s + q_chunk, L)
                # Round kv to 4K blocks: few distinct buffer sizes let the
                # allocator reuse cache (per-chunk sizes would otherwise sum to
                # ~L^2/2 of dead cache). The causal mask zeroes the padded tail.
                kv_end = min(((e + 4095) // 4096) * 4096, L) if causal else L
                q_c = qf[:, :, :, s:e]
                if outs:  # chain chunks so transients are live one at a time
                    q_c = mx.depends([q_c], [outs[-1]])[0]
                sc = scale * (q_c @ kf[:, :, :, :kv_end].swapaxes(-1, -2))
                if causal:
                    rows = mx.arange(s, e)[:, None]
                    cols = mx.arange(kv_end)[None, :]
                    neg = mx.array(
                        -65504.0 if sc.dtype == mx.float16 else -1e30, sc.dtype
                    )
                    sc = mx.where(cols <= rows, sc, neg)
                p = mx.softmax(sc, axis=-1, precise=True).astype(qq.dtype)
                outs.append(p @ vf[:, :, :, :kv_end])
            return mx.concatenate(outs, axis=3).reshape(B, Hq, L, Dh)

        @flash.vjp
        def flash_vjp(primals, cotangents, outputs):
            global _chunked_chain
            qq, kk, vv = primals
            do = cotangents if isinstance(cotangents, mx.array) else cotangents[0]
            B, Hq, L, Dh = qq.shape
            Hk = kk.shape[1]
            rep = Hq // Hk

            f32 = mx.float32
            wt = qq.dtype  # compute in input dtype; softmax precise (fp32 inside)
            neg = mx.array(-65504.0 if wt == mx.float16 else -1e30, wt)
            # GQA via broadcast (no repeated K/V copies): q as (B,Hk,rep,L,D),
            # k/v as (B,Hk,1,L,D).
            qf = qq.reshape(B, Hk, rep, L, Dh)
            kf = kk[:, :, None]
            vf = vv[:, :, None]
            dof = do.reshape(B, Hk, rep, L, Dh).astype(wt)

            dq_l = []
            dk = mx.zeros((B, Hk, 1, L, Dh), dtype=f32)
            dv = mx.zeros((B, Hk, 1, L, Dh), dtype=f32)
            # Gate this layer's first chunk on the previous attention layer's
            # finished accumulators, so layer backwards serialize instead of
            # overlapping under checkpoint recompute.
            prev = _chunked_chain
            if prev is not None:
                dk, dv = mx.depends([dk, dv], prev)
            for s in range(0, L, q_chunk):
                e = min(s + q_chunk, L)
                kv_end = min(((e + 4095) // 4096) * 4096, L) if causal else L
                # Gate this chunk's recompute on the accumulator chain so chunk
                # working sets are live one at a time.
                q_c, k_a, v_a = mx.depends(
                    [qf[:, :, :, s:e], kf[:, :, :, :kv_end], vf[:, :, :, :kv_end]],
                    [dk, dv],
                )
                sc = (scale * (q_c @ k_a.swapaxes(-1, -2))).astype(wt)
                if causal:
                    rows = mx.arange(s, e)[:, None]
                    cols = mx.arange(kv_end)[None, :]
                    sc = mx.where(cols <= rows, sc, neg)
                p = mx.softmax(sc, axis=-1, precise=True)
                do_c = dof[:, :, :, s:e]
                dp = do_c @ v_a.swapaxes(-1, -2)
                ds = p * (dp - (dp * p).sum(-1, keepdims=True)) * scale
                dq_c = ds @ k_a  # broadcast over rep
                dk_c = (ds.swapaxes(-1, -2) @ q_c).sum(2, keepdims=True).astype(f32)
                dv_c = (p.swapaxes(-1, -2) @ do_c).sum(2, keepdims=True).astype(f32)
                pad = L - kv_end
                if pad:
                    dk_c = mx.pad(dk_c, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
                    dv_c = mx.pad(dv_c, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
                dk = dk + dk_c
                dv = dv + dv_c
                dq_l.append(dq_c)

            dq = mx.concatenate(dq_l, axis=3)
            _chunked_chain = [dk, dv, dq]
            return [
                dq.reshape(B, Hq, L, Dh).astype(qq.dtype),
                dk[:, :, 0].astype(kk.dtype),
                dv[:, :, 0].astype(vv.dtype),
            ]

        return flash(q, k, v)

    return sdpa


def set_chunked_attention(q_chunk: int, min_len: int = 1024) -> None:
    """Enable (q_chunk > 0) or disable (0) the chunked-backward attention path.

    Called by the trainer when --chunked-attention-q is set; off otherwise.
    """
    global _CHUNKED_ATTN_Q, _CHUNKED_ATTN_MIN_LEN, _chunked_sdpa
    _CHUNKED_ATTN_Q = q_chunk
    _CHUNKED_ATTN_MIN_LEN = min_len
    _chunked_sdpa = make_chunked_sdpa(q_chunk) if q_chunk > 0 else None


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        # Opt-in chunked backward for long, full-sequence causal training only
        # (cache is non-quantized here). Default _CHUNKED_ATTN_Q == 0 -> stock.
        if _CHUNKED_ATTN_Q and _chunked_sdpa is not None:
            long_causal = (
                (mask == "causal" or mask is None)
                and sinks is None
                and queries.shape[2] >= _CHUNKED_ATTN_MIN_LEN
                and queries.shape[2] == keys.shape[2]  # full-sequence (training)
            )
            if long_causal:
                return _chunked_sdpa(queries, keys, values, scale=scale, mask=mask)
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
