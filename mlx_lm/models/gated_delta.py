from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


# ---------------------------------------------------------------------------
# Opt-in analytic chunkwise backward for the gated-delta scan (training only).
#
# Stock training (use_kernel=False) differentiates through gated_delta_ops, whose
# per-step autodiff retains the scan's VJP primals for the whole sequence -- the
# dominant training-memory term at long context. When enabled (see
# set_analytic_gated_delta), the scan runs in chunks under a hand-derived custom
# VJP: per chunk a short recompute, batched tensor ops, and a single state-sized
# reverse carrier, so chunk working sets are live one at a time. The forward is
# bit-identical (delegates to gated_delta_ops); gradients agree to floating-point
# tolerance. Default off; inference is never affected.
# ---------------------------------------------------------------------------

_ANALYTIC_CHUNK_STEPS = None


def set_analytic_gated_delta(chunk_steps):
    """Enable (``chunk_steps`` > 0) or disable (``None``/0) the analytic chunkwise
    backward used during training. Inference is unaffected."""
    global _ANALYTIC_CHUNK_STEPS
    _ANALYTIC_CHUNK_STEPS = chunk_steps if chunk_steps else None


def _make_gated_delta_bwd_kernel(has_mask):
    """Fused backward kernel: pass 1 writes the pre-step state tape, pass 2 runs
    the gradient carrier in registers and writes the G' tape. Rows (b, hv, dv)
    are independent end to end, so the two passes need no synchronization. Masked
    steps mirror gated_delta_ops (state kept; the y/dy path stays active)."""
    if not mx.metal.is_available():
        return None
    mask_src = "mask[b_idx * T + t]" if has_mask else "true"
    mask_src_r = "mask[b_idx * T + tr]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto dy_ = dy + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto d_sout_ = d_sout + (n * Dv + dv_idx) * Dk;
        auto d_sin_ = d_sin + (n * Dv + dv_idx) * Dk;
        // tapes: [B, T, Hv, Dv, Dk]
        auto sp_ = sprev_tape + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk;
        auto gp_ = gp_tape + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk;

        float st[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          st[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }}

        // Pass 1: forward recurrence, taping pre-step states.
        for (int t = 0; t < T; ++t) {{
          for (int i = 0; i < n_per_t; ++i) {{
            sp_[n_per_t * dk_idx + i] = st[i];
          }}
          if ({mask_src}) {{
            float kv_mem = 0.0f;
            float tmp[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              tmp[i] = st[i] * g_[hv_idx];
              kv_mem += tmp[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);
            auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];
            for (int i = 0; i < n_per_t; ++i) {{
              st[i] = tmp[i] + k_[n_per_t * dk_idx + i] * delta;
            }}
          }}
          q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv;
          g_ += Hv; beta_ += Hv;
          sp_ += Hv * Dv * Dk;
        }}

        // Pass 2: reverse carrier; emit G' tape.
        float c[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          c[i] = static_cast<float>(d_sout_[n_per_t * dk_idx + i]);
        }}
        auto qr = q + (b_idx * T + (T - 1)) * Hk * Dk + hk_idx * Dk;
        auto kr = k + (b_idx * T + (T - 1)) * Hk * Dk + hk_idx * Dk;
        auto dyr = dy + (b_idx * T + (T - 1)) * Hv * Dv + hv_idx * Dv;
        auto gr = g + (b_idx * T + (T - 1)) * Hv;
        auto br = beta + (b_idx * T + (T - 1)) * Hv;
        gp_ += (T - 1) * Hv * Dv * Dk;
        for (int tr = T - 1; tr >= 0; --tr) {{
          float dyv = static_cast<float>(dyr[dv_idx]);
          bool live = {mask_src_r};
          float gp[n_per_t];
          float u = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            gp[i] = (live ? c[i] : 0.0f) + dyv * static_cast<float>(qr[s_idx]);
            u += gp[i] * static_cast<float>(kr[s_idx]);
          }}
          u = simd_sum(u);
          float dm = -br[hv_idx] * u;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            gp_[s_idx] = gp[i];
            float cn = gr[hv_idx] * (gp[i] + dm * static_cast<float>(kr[s_idx]));
            c[i] = live ? cn : (cn + c[i]);
          }}
          qr -= Hk * Dk; kr -= Hk * Dk; dyr -= Hv * Dv;
          gr -= Hv; br -= Hv;
          gp_ -= Hv * Dv * Dk;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          d_sin_[n_per_t * dk_idx + i] = c[i];
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "dy", "d_sout", "T"]
    if has_mask:
        inputs.append("mask")
    return mx.fast.metal_kernel(
        name=f"gated_delta_bwd{'_mask' if has_mask else ''}",
        input_names=inputs,
        output_names=["sprev_tape", "gp_tape", "d_sin"],
        source=source,
    )


_gated_delta_bwd_kernel = _make_gated_delta_bwd_kernel(False)
_gated_delta_bwd_kernel_masked = _make_gated_delta_bwd_kernel(True)


def _analytic_chunk_vjp(primals, cotangents, mask_c):
    # Gate the whole chunk backward on its cotangents: the scheduler cannot start
    # this chunk's recompute before the grad chain reaches it, so chunk working
    # sets are live one at a time instead of piling across chunks.
    primals = mx.depends(list(primals), list(cotangents))
    state_in, q, k, v, g, beta = primals
    dy, d_sout = cotangents
    B, C, Hk, Dk = q.shape
    Hv, _Dv = v.shape[-2:]
    rep = Hv // Hk

    f32 = mx.float32
    qe = mx.repeat(q, rep, -2).astype(f32) if rep > 1 else q.astype(f32)
    ke = mx.repeat(k, rep, -2).astype(f32) if rep > 1 else k.astype(f32)
    vf, gf, bf = v.astype(f32), g.astype(f32), beta.astype(f32)
    dy = dy.astype(f32)
    c = d_sout.astype(f32)

    use_kernel = (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and Dk % 32 == 0
        and _Dv % 4 == 0
        and _gated_delta_bwd_kernel is not None
    )
    if use_kernel:
        kern = (
            _gated_delta_bwd_kernel_masked
            if mask_c is not None
            else _gated_delta_bwd_kernel
        )
        ins = [qe, ke, vf, gf, bf, state_in.astype(f32), dy, c, C]
        if mask_c is not None:
            ins.append(mask_c)
        s_prev, gp_tape, c_out = kern(
            inputs=ins,
            template=[
                ("InT", qe.dtype),
                ("StT", f32),
                ("Dk", Dk),
                ("Dv", _Dv),
                ("Hk", Hv),  # qe/ke already expanded to Hv heads
                ("Hv", Hv),
            ],
            grid=(32, _Dv, B * Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(B, C, Hv, _Dv, Dk), (B, C, Hv, _Dv, Dk), (B, Hv, _Dv, Dk)],
            output_dtypes=[f32, f32, f32],
        )
    else:
        # Ops fallback: recompute trajectory + reverse carrier in Python.
        states = [state_in.astype(f32)]
        for t in range(C):
            _, s_next = _gated_delta_step_ops(
                qe[:, t],
                ke[:, t],
                vf[:, t],
                gf[:, t],
                bf[:, t],
                states[-1],
                None if mask_c is None else mask_c[:, t],
            )
            states.append(s_next)
        s_prev = mx.stack(states[:-1], axis=1)

    g_ = gf[..., None, None]  # (B, C, Hv, 1, 1)
    b_ = bf[..., None]  # (B, C, Hv, 1)
    s_prm = (
        g_ * s_prev
        + (b_ * (vf - ((g_ * s_prev) * ke[..., None, :]).sum(-1)))[..., None]
        * ke[..., None, :]
    )  # S'
    m = ((g_ * s_prev) * ke[..., None, :]).sum(-1)  # (B, C, Hv, Dv)
    d = b_ * (vf - m)  # delta

    dq_e = (dy[..., None] * s_prm).sum(-2)  # (B, C, Hv, Dk)

    if not use_kernel:
        gp_l = []
        for t in range(C - 1, -1, -1):
            if mask_c is None:
                gp = c + dy[:, t][..., None] * qe[:, t][..., None, :]
                pass_th = None
            else:
                m_t = mask_c[:, t][..., None, None, None]
                gp = (
                    mx.where(m_t, c, 0.0) + dy[:, t][..., None] * qe[:, t][..., None, :]
                )
                pass_th = mx.where(m_t, mx.array(0.0, f32), c)
            u = (gp * ke[:, t][..., None, :]).sum(-1)
            dm = -bf[:, t][..., None] * u
            c = gf[:, t][..., None, None] * (
                gp + dm[..., None] * ke[:, t][..., None, :]
            )
            if pass_th is not None:
                c = c + pass_th
            gp_l.append(gp)
        gp_tape = mx.stack(gp_l[::-1], axis=1)
        c_out = c

    u_b = (gp_tape * ke[..., None, :]).sum(-1)  # (B, C, Hv, Dv)
    dm_b = -b_ * u_b
    dv = b_ * u_b
    db = (u_b * (vf - m)).sum(-1)
    dk_e = (gp_tape * d[..., None]).sum(-2) + ((g_ * s_prev) * dm_b[..., None]).sum(-2)
    ds_tld = gp_tape + dm_b[..., None] * ke[..., None, :]
    dg = (ds_tld * s_prev).sum((-1, -2))

    if rep > 1:
        dq_e = dq_e.reshape(B, C, Hk, rep, Dk).sum(3)
        dk_e = dk_e.reshape(B, C, Hk, rep, Dk).sum(3)

    # Chain the carrier on this chunk's finished gradients so the next (earlier)
    # chunk cannot start until these tapes were consumed.
    c_out = mx.depends([c_out], [dq_e, dk_e, dv, dg, db])[0]
    return [
        c_out.astype(state_in.dtype),
        dq_e.astype(q.dtype),
        dk_e.astype(k.dtype),
        dv.astype(v.dtype),
        dg.astype(g.dtype),
        db.astype(beta.dtype),
    ]


def _make_analytic_chunk_op(mask_c):
    """Custom-VJP chunk op; mask rides in the closure (no gradient needed)."""

    @mx.custom_function
    def chunk_op(state_in, q, k, v, g, beta):
        return gated_delta_ops(q, k, v, g, beta, state_in, mask_c)

    @chunk_op.vjp
    def chunk_op_vjp(primals, cotangents, outputs):
        return _analytic_chunk_vjp(list(primals), list(cotangents), mask_c)

    return chunk_op


def _analytic_gated_delta_scan(q, k, v, g, beta, state, mask, chunk_steps):
    B, T, _, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    ys = []
    for s in range(0, T, chunk_steps):
        e = min(s + chunk_steps, T)
        op = _make_analytic_chunk_op(None if mask is None else mask[:, s:e])
        y_c, state = op(state, q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e])
        ys.append(y_c)
    return mx.concatenate(ys, axis=1), state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        # Training path: the analytic chunkwise backward when enabled (and the
        # gating is scalar-per-head, the derived case); otherwise the stock scan.
        if _ANALYTIC_CHUNK_STEPS and g.ndim == 3:
            return _analytic_gated_delta_scan(
                q, k, v, g, beta, state, mask, _ANALYTIC_CHUNK_STEPS
            )
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
