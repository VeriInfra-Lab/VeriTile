import VeriTile.Triton.DSL.RebindingProcedures
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Complete repaired Python bodies. The invocation supplies FP32 storage,
scalar types and constexpr values outside each untouched source body. -/
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def forwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("o",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def forwardPastedKernel
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
    (BTL BTS BK BV DK DV : Nat) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, B := B, H := H, T := T, scale := scale]
    rebinding_jit_procedures [] in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_kv, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), ((i_c * BTL).to(tl.int32), (i_k * BK).to(tl.int32)), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), ((i_k * BK).to(tl.int32), 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, (i_v * BV).to(tl.int32)), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False) * d_h[None, :]
        # [BQ, BD]
        b_o = b_o * tl.math.exp2(b_b * BTS)
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    d_q = tl.math.exp2(tl.arange(0, BTL) * b_b)
    b_o *= d_q[:, None]
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), ((i_k * BK).to(tl.int32), (i_c * BTL).to(tl.int32)), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), ((i_c * BTL).to(tl.int32), (i_v * BV).to(tl.int32)), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2(
            (o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), ((i_c*BTL).to(tl.int32), (i_v*BV).to(tl.int32)), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def querySignature : CallSignature where
  pointers := [("k",.fp32),("v",.fp32),("do",.fp32),("dq",.fp32)]
  scalars := [("i_bh",.integer),("i_c",.integer),("i_k",.integer),("i_v",.integer),("i_h",.integer),("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def queryProcedureSource : ProcedureJitFunction :=
  with_triton_params (i_bh,i_c,i_k,i_v,i_h,k,v,«do»,dq,s_qk_h,s_qk_t,s_qk_d,s_vo_h,s_vo_t,s_vo_d,B,H,T,scale,BTL,BTS,BK,BV,DK,DV) constexpr_args [BTL,BTS,BK,BV,DK,DV] rebinding_signature querySignature in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_bh = tl.cast(i_bh, tl.int64)
    i_c = tl.cast(i_c, tl.int64)
    i_k = tl.cast(i_k, tl.int64)
    i_v = tl.cast(i_v, tl.int64)
    i_h = tl.cast(i_h, tl.int64)
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d),
                             ((i_c * BTL).to(tl.int32), (i_v * BV).to(tl.int32)), (BTL, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (0, (i_k * BK).to(tl.int32)), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T),
                            (s_vo_d, s_vo_t), ((i_v * BV).to(tl.int32), 0), (BV, BTS), (0, 1))
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # overall decay rate for an entire block
    d_b = tl.math.exp2(b_b * BTS)
    # cumulative decay from the end of the chunk
    d_h = tl.math.exp2((BTS - tl.arange(0, BTS)) * b_b)
    for _ in range(0, i_c * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_h[None, :]
        # [BQ, BD]
        b_dq *= d_b
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
    b_dq *= tl.math.exp2(tl.arange(0, BTL) * b_b)[:, None] * scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), ((i_c * BTL).to(tl.int32), (i_k * BK).to(tl.int32)), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T),
                            (s_vo_d, s_vo_t), ((i_v * BV).to(tl.int32), (i_c * BTL).to(tl.int32)), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2(
            (o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s * scale
        # [BTL, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, DK),
                             (s_qk_t, s_qk_d), ((i_c*BTL).to(tl.int32), (i_k*BK).to(tl.int32)), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return
  }

def keyValueSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("i_bh",.integer),("i_c",.integer),("i_k",.integer),("i_v",.integer),("i_h",.integer),("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def keyValueProcedureSource : ProcedureJitFunction :=
  with_triton_params (i_bh,i_c,i_k,i_v,i_h,q,k,v,«do»,dk,dv,s_qk_h,s_qk_t,s_qk_d,s_vo_h,s_vo_t,s_vo_d,B,H,T,scale,BTL,BTS,BK,BV,DK,DV) constexpr_args [BTL,BTS,BK,BV,DK,DV] rebinding_signature keyValueSignature in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_bh = tl.cast(i_bh, tl.int64)
    i_c = tl.cast(i_c, tl.int64)
    i_k = tl.cast(i_k, tl.int64)
    i_v = tl.cast(i_v, tl.int64)
    i_h = tl.cast(i_h, tl.int64)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # overall decay rate for an entire block
    d_b = tl.math.exp2(b_b * BTS)
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d),
                            ((i_c * BTL).to(tl.int32), (i_k * BK).to(tl.int32)), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d),
                            ((i_c * BTL).to(tl.int32), (i_v * BV).to(tl.int32)), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1), padding_option="zero"), tl.load(
        p_v, boundary_check=(0, 1), padding_option="zero")
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32)
    d_h = tl.math.exp2((BTL - tl.arange(0, BTL)) * b_b)
    b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
    d_q = tl.math.exp2(tl.arange(0, BTS) * b_b)
    for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), ((i_k * BK).to(tl.int32), tl.cast(i, tl.int32)), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), ((i_v * BV).to(tl.int32), tl.cast(i, tl.int32)), (BV, BTS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")  # [BK, BTS]
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")  # [BV, BTS]
        b_do = (b_do * d_q[None, :]).to(b_do.dtype)

        b_dv *= d_b
        b_s = tl.dot(b_kd.to(b_q.dtype), b_q, allow_tf32=False)  # [BTL, BTS]
        b_dv += tl.dot(b_s.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)

        b_dk *= d_b
        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        b_dk += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
    b_dk *= d_h[:, None] * scale
    b_dv *= scale
    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), ((i_k * BK).to(tl.int32), tl.cast(i, tl.int32)), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), ((i_v * BV).to(tl.int32), tl.cast(i, tl.int32)), (BV, BTS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")  # [BD, BQ]
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        # [BK, BQ]
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2(
            (-o_k[:, None] + o_q[None, :]) * b_b.to(tl.float32)), 0) * scale
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * d_s
        # [BK, BD]
        b_dk += tl.dot(b_ds.to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        o_q += BTS
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h,
                             (T, DK), (s_qk_t, s_qk_d), ((i_c*BTL).to(tl.int32), (i_k*BK).to(tl.int32)), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h,
                             (T, DV), (s_vo_t, s_vo_d), ((i_c*BTL).to(tl.int32), (i_v*BV).to(tl.int32)), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return
  }

def backwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def backwardPastedKernel
    (q k v «do» dq dk dv : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
    (BTL BTS BK BV DK DV : Nat) : ComputeKernel :=
  with_triton_call backwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, B := B, H := H, T := T, scale := scale]
    rebinding_jit_procedures [_parallel_retention_bwd_dq := queryProcedureSource, _parallel_retention_bwd_dkv := keyValueProcedureSource] in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_kv, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_retention_bwd_dq(
        i_bh, i_c, i_k, i_v, i_h,
        k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, B, H, T, scale,  BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV
    )
    tl.debug_barrier()
    _parallel_retention_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, DK, DV
    )
  }

def fusedForwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("o",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def fusedForwardPastedKernel
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
    (BTL BTS BK BV DK DV : Nat) : ComputeKernel :=
  with_triton_call fusedForwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, B := B, H := H, T := T, scale := scale]
    rebinding_jit_procedures [] in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_kv, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # cumulative decay from the end of the chunk
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), ((i_c * BTL).to(tl.int32), (i_k * BK).to(tl.int32)), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), ((i_k * BK).to(tl.int32), 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, (i_v * BV).to(tl.int32)), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False) * d_h[None, :]
        # [BQ, BD]
        b_o = b_o * tl.math.exp2(b_b * BTS) + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    d_q = tl.math.exp2(tl.arange(0, BTL) * b_b)
    b_o *= d_q[:, None]
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), ((i_k * BK).to(tl.int32), (i_c * BTL).to(tl.int32)), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), ((i_c * BTL).to(tl.int32), (i_v * BV).to(tl.int32)), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2(
            (o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), ((i_c*BTL).to(tl.int32), (i_v*BV).to(tl.int32)), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def fusedQuerySignature : CallSignature where
  pointers := [("k",.fp32),("v",.fp32),("do",.fp32),("dq",.fp32)]
  scalars := [("i_bh",.integer),("i_c",.integer),("i_k",.integer),("i_v",.integer),("i_h",.integer),("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def fusedQueryProcedureSource : ProcedureJitFunction :=
  with_triton_params (i_bh,i_c,i_k,i_v,i_h,k,v,«do»,dq,s_qk_h,s_qk_t,s_qk_d,s_vo_h,s_vo_t,s_vo_d,B,H,T,scale,BTL,BTS,BK,BV,DK,DV) constexpr_args [BTL,BTS,BK,BV,DK,DV] rebinding_signature fusedQuerySignature in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_bh = tl.cast(i_bh, tl.int64)
    i_c = tl.cast(i_c, tl.int64)
    i_k = tl.cast(i_k, tl.int64)
    i_v = tl.cast(i_v, tl.int64)
    i_h = tl.cast(i_h, tl.int64)
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d),
                             ((i_c * BTL).to(tl.int32), (i_v * BV).to(tl.int32)), (BTL, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (0, (i_k * BK).to(tl.int32)), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T),
                            (s_vo_d, s_vo_t), ((i_v * BV).to(tl.int32), 0), (BV, BTS), (0, 1))
    # decay rate given the head index
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    # overall decay rate for an entire block
    d_b = tl.math.exp2(b_b * BTS)
    # cumulative decay from the end of the chunk
    d_h = tl.math.exp2((BTS - tl.arange(0, BTS)) * b_b)
    for _ in range(0, i_c * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_h[None, :]
        # [BQ, BD]
        b_dq = b_dq * d_b + tl.dot(b_ds.to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
    b_dq *= tl.math.exp2(tl.arange(0, BTL) * b_b)[:, None] * scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), ((i_c * BTL).to(tl.int32), (i_k * BK).to(tl.int32)), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T),
                            (s_vo_d, s_vo_t), ((i_v * BV).to(tl.int32), (i_c * BTL).to(tl.int32)), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2(
            (o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s * scale
        # [BTL, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, DK),
                             (s_qk_t, s_qk_d), ((i_c*BTL).to(tl.int32), (i_k*BK).to(tl.int32)), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return
  }

def fusedBackwardSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("do",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("B",.integer),("H",.integer),("T",.integer),("scale",.fp32)]
  constexpr := ["BTL","BTS","BK","BV","DK","DV"]

def fusedBackwardPastedKernel
    (q k v «do» dq dk dv : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
    (BTL BTS BK BV DK DV : Nat) : ComputeKernel :=
  with_triton_call fusedBackwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, B := B, H := H, T := T, scale := scale]
    rebinding_jit_procedures [_parallel_retention_bwd_dq_fused := fusedQueryProcedureSource, _parallel_retention_bwd_dkv := keyValueProcedureSource] in triton {
    s_qk_h = tl.cast(s_qk_h, tl.int64)
    s_qk_t = tl.cast(s_qk_t, tl.int64)
    s_qk_d = tl.cast(s_qk_d, tl.int64)
    s_vo_h = tl.cast(s_vo_h, tl.int64)
    s_vo_t = tl.cast(s_vo_t, tl.int64)
    s_vo_d = tl.cast(s_vo_d, tl.int64)
    B = tl.cast(B, tl.int64)
    H = tl.cast(H, tl.int64)
    T = tl.cast(T, tl.int64)
    i_kv, i_c, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_retention_bwd_dq_fused(
        i_bh, i_c, i_k, i_v, i_h,
        k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, B, H, T, scale,  BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV
    )
    tl.debug_barrier()
    _parallel_retention_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, DK, DV
    )
  }

end
attribute [reducible] forwardPastedKernel backwardPastedKernel fusedForwardPastedKernel fusedBackwardPastedKernel
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
