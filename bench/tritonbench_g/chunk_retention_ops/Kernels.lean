import VeriTile.Triton.DSL.SignedArange
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete repaired Python function bodies with the existing scale-first
candidate. FP32 storage and contiguous strides come from caller metadata;
all source casts, signed lanes, tail branches, loop directions and stores remain. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.Source
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000

def forwardHSignature : CallSignature where
  pointers := [("k",.fp32),("v",.fp32),("h",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer)]
  constexpr := ["H","T","K","V","BT","BK","BV","NT","USE_INITIAL_STATE","STORE_FINAL_STATE"]
  readPorts := ["k","v","h0"]
  writePorts := ["h","ht"]

def forwardHKernel (k v h h0 ht : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  with_triton_call forwardHSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t] in triton {
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        if i_t == NT - 1 and (T % BT) != 0:
            d_b = tl.math.exp2((T % BT) * b_b)
            d_i = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
  }

def forwardOSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("h",.fp32),("v",.fp32),("o",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["H","T","K","V","BT","BK","BV"]
  readPorts := ["q","k","v","h"]
  writePorts := ["o"]

def forwardOKernel (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat) : ComputeKernel :=
  with_triton_call forwardOSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s *= d_s
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def forwardOScaleFirstKernel (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat) : ComputeKernel :=
  with_triton_call forwardOSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s *= d_s
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = scale * (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def backwardQKVSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("h",.fp32),("do",.fp32),("dh",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["H","T","K","V","BT","BK","BV","NT"]
  readPorts := ["q","k","v","h","do","dh"]
  writePorts := ["dq","dk","dv"]

def backwardQKVKernel (q k v h «do» dh dq dk dv : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat) : ComputeKernel :=
  with_triton_call backwardQKVSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    n_bh = tl.num_programs(2).to(tl.int64)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    actual_length = tl.minimum(BT, T - i_t * BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((actual_length - o_i - 1) * b_b)
    d_q = (d_q * scale).to(d_q.dtype)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    b_ds = (b_ds * d_s).to(b_q.dtype)
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
  }

def backwardHSignature : CallSignature where
  pointers := [("q",.fp32),("do",.fp32),("dh",.fp32),("d_ht",.fp32),("d_h0",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["H","T","K","V","BT","BK","BV","NT","USE_FINAL_GRAD","STORE_INITIAL_GRAD"]
  readPorts := ["q","do","d_ht"]
  writePorts := ["dh","d_h0"]

def backwardHKernel (q «do» dh d_ht d_h0 : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat) (USE_FINAL_GRAD STORE_INITIAL_GRAD : Bool) : ComputeKernel :=
  with_triton_call backwardHSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRAD:
        p_dht = tl.make_block_ptr(d_ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dht, boundary_check=(0, 1), padding_option="zero")
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        actual_length = tl.minimum(BT, T - i_t * BT)
        d_b = tl.math.exp2(actual_length * b_b)
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=False)
    if STORE_INITIAL_GRAD:
        p_dh0 = tl.make_block_ptr(d_h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
  }

end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.Source
