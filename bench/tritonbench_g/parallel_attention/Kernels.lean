import VeriTile.Triton.DSL.ProcedureJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Authoritative repaired Python function bodies, including helper calls,
empty helper returns, source casts, stores and descending ranges. -/
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def forwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("o",.fp32),("z",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("scale",.fp32),("B",.integer),("H",.integer),("T",.integer)]
  constexpr := ["K","V","BTL","BTS","BK","BV"]

def forwardPastedKernel
    (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
    (K V BTL BTS BK BV : Nat) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, scale := scale, B := B, H := H, T := T] in triton {
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_z, b_z.to(p_z.dtype.element_ty),
                 mask=((i_c * BTL + tl.arange(0, BTL)) < T))
  }

def querySignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dz",.fp32),("dq",.fp32)]
  scalars := [("i_bh",.integer),("i_c",.integer),("i_k",.integer),("i_v",.integer),("i_h",.integer),("s_k_h",.integer),("s_k_t",.integer),("s_k_d",.integer),("s_v_h",.integer),("s_v_t",.integer),("s_v_d",.integer),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BTL","BTS","BK","BV"]

def queryProcedureSource : ProcedureJitFunction :=
  with_triton_params (i_bh,i_c,i_k,i_v,i_h,q,k,v,«do»,dz,dq,s_k_h,s_k_t,s_k_d,s_v_h,s_v_t,s_v_d,scale,B,H,T,K,V,BTL,BTS,BK,BV) constexpr_args [B,H,T,K,V,BTL,BTS,BK,BV] signature querySignature in triton {
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                             (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c * BTL + tl.arange(0, BTL)) < T, other=0)

    for _ in range(0, i_c * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        # [BQ, BD]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BTL, BK]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_k_h, (T, K),
                             (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return
  }

def keyValueSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dz",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("i_bh",.integer),("i_c",.integer),("i_k",.integer),("i_v",.integer),("i_h",.integer),("s_k_h",.integer),("s_k_t",.integer),("s_k_d",.integer),("s_v_h",.integer),("s_v_t",.integer),("s_v_d",.integer),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BTL","BTS","BK","BV"]

def keyValueSource : ProcedureJitFunction :=
  with_triton_params (i_bh,i_c,i_k,i_v,i_h,q,k,v,«do»,dz,dk,dv,s_k_h,s_k_t,s_k_d,s_v_h,s_v_t,s_v_d,scale,B,H,T,K,V,BTL,BTS,BK,BV) constexpr_args [B,H,T,K,V,BTL,BTS,BK,BV] signature keyValueSignature in triton {
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                            (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1), padding_option="zero"), tl.load(
        p_v, boundary_check=(0, 1), padding_option="zero")
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32)

    for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")  # [BK, BTS]
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)  # [BV, BTS]
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T, other=0)
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * \
            scale  # [BTL, BTS]
        # Padded future query lanes must not contribute even for non-finite values.
        valid_query = (i + tl.arange(0, BTS)) < T
        b_s = tl.where(valid_query[None, :], b_s, 0)
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_ds = tl.where(valid_query[None, :], b_ds, 0)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype),
                       tl.trans(b_q), allow_tf32=False)

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")  # [BD, BQ]
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T, other=0)
        # [BK, BQ]
        m_s = (o_k[:, None] <= o_q[None, :]) & ((i + tl.arange(0, BTS))[None, :] < T)
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        # [BK, BD]
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype),
                       tl.trans(b_q), allow_tf32=False)
        o_q += BTS

    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_k_h,
                             (T, K), (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_v_h,
                             (T, V), (s_v_t, s_v_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return
  }

def backwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dz",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BTL","BTS","BK","BV"]

def backwardPastedKernel
    (q k v «do» dz dq dk dv : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  with_triton_call backwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, scale := scale]
    jit_procedures [_parallel_rebased_bwd_dq := queryProcedureSource, _parallel_rebased_bwd_dkv := keyValueSource] in triton {
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d,
        scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )
  }

def fusedForwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("o",.fp32),("z",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("scale",.fp32),("B",.integer),("H",.integer),("T",.integer)]
  constexpr := ["K","V","BTL","BTS","BK","BV"]

def fusedForwardKernel
    (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
    (K V BTL BTS BK BV : Nat) : ComputeKernel :=
  with_triton_call fusedForwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, scale := scale, B := B, H := H, T := T] in triton {
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s * b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_z, b_z.to(p_z.dtype.element_ty),
                 mask=((i_c * BTL + tl.arange(0, BTL)) < T))
  }

def fusedQuerySignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dz",.fp32),("dq",.fp32)]
  scalars := [("i_bh",.integer),("i_c",.integer),("i_k",.integer),("i_v",.integer),("i_h",.integer),("s_k_h",.integer),("s_k_t",.integer),("s_k_d",.integer),("s_v_h",.integer),("s_v_t",.integer),("s_v_d",.integer),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BTL","BTS","BK","BV"]

def fusedQueryProcedureSource : ProcedureJitFunction :=
  with_triton_params (i_bh,i_c,i_k,i_v,i_h,q,k,v,«do»,dz,dq,s_k_h,s_k_t,s_k_d,s_v_h,s_v_t,s_v_d,scale,B,H,T,K,V,BTL,BTS,BK,BV) constexpr_args [B,H,T,K,V,BTL,BTS,BK,BV] signature fusedQuerySignature in triton {
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                             (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c * BTL + tl.arange(0, BTL)) < T, other=0)

    for _ in range(0, i_c * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        # [BQ, BD]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.where(m_s, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0)
        # [BTL, BK]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_k_h, (T, K),
                             (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return
  }

def fusedBackwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dz",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("scale",.fp32)]
  constexpr := ["B","H","T","K","V","BTL","BTS","BK","BV"]

def fusedBackwardKernel
    (q k v «do» dz dq dk dv : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  with_triton_call fusedBackwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, scale := scale]
    jit_procedures [_parallel_rebased_bwd_dq_fused := fusedQueryProcedureSource, _parallel_rebased_bwd_dkv := keyValueSource] in triton {
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq_fused(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d,
        scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )
  }

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel

attribute [reducible] VeriTile.Bench.Repairs.ParallelAttentionKernel.forwardPastedKernel
  VeriTile.Bench.Repairs.ParallelAttentionKernel.fusedForwardKernel
  VeriTile.Bench.Repairs.ParallelAttentionKernel.backwardPastedKernel
  VeriTile.Bench.Repairs.ParallelAttentionKernel.fusedBackwardKernel
