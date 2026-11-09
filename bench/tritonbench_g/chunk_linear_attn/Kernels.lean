import VeriTile.Triton.DSL.PythonShapeExpressions
import VeriTile.Triton.DSL.PastedRangeLoads
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile.Triton
noncomputable section
structure Parameters where
  (B H T K V BT BK BV NT : Nat)

/-! Complete repaired Python function bodies. Caller metadata supplies the
selected FP32 storage and exact contiguous strides. Every source cast, tuple,
loop direction, branch, mask and store remains in the pasted bodies. -/
section
def forwardHSignature : CallSignature where
  pointers := [("k",.fp32),("v",.fp32),("h",.fp32),("h0",.fp32),("ht",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer)]
  constexpr := ["T","K","V","BT","BK","BV","NT","USE_INITIAL_STATE","STORE_FINAL_STATE"]
  readPorts := ["k","v","h0"]
  writePorts := ["h","ht"]

def forwardOSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("h",.fp32),("v",.fp32),("o",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","V","BT","BK","BV"]
  readPorts := ["q","k","h","v"]
  writePorts := ["o"]

def backwardHSignature : CallSignature where
  pointers := [("q",.fp32),("do",.fp32),("dh",.fp32),("dht",.fp32),("dh0",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","V","BT","BK","BV","NT","USE_FINAL_GRADIENT","STORE_INITIAL_GRADIENT"]
  readPorts := ["q","do","dht"]
  writePorts := ["dh","dh0"]

def backwardQKVSignature : CallSignature where
  pointers := [("q",.fp32),("k",.fp32),("v",.fp32),("h",.fp32),("do",.fp32),("dh",.fp32),("dq",.fp32),("dk",.fp32),("dv",.fp32)]
  scalars := [("s_qk_h",.integer),("s_qk_t",.integer),("s_qk_d",.integer),("s_vo_h",.integer),("s_vo_t",.integer),("s_vo_d",.integer),("s_h_h",.integer),("s_h_t",.integer),("scale",.fp32)]
  constexpr := ["T","K","V","BT","BK","BV","NT"]
  readPorts := ["q","k","v","h","do","dh"]
  writePorts := ["dq","dk","dv"]

def forwardHKernel (p : Parameters) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let k : RegionName := "key"
  let v : RegionName := "value"
  let h : RegionName := "history"
  let h0 : RegionName := "initial"
  let ht : RegionName := "final"
  with_triton_call forwardHSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t] in triton {
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

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
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
  }

def forwardOKernel (p : Parameters) (scale : ℝ) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let k : RegionName := "key"
  let h : RegionName := "history"
  let v : RegionName := "value"
  let o : RegionName := "output"
  with_triton_call forwardOSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

def backwardHKernel (p : Parameters) (scale : ℝ) (USE_FINAL_GRADIENT STORE_INITIAL_GRADIENT : Bool) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let «do» : RegionName := "incoming"
  let dh : RegionName := "historyGradient"
  let dht : RegionName := "terminalGradient"
  let dh0 : RegionName := "initialGradient"
  with_triton_call backwardHSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dht, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)

    if STORE_INITIAL_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
  }

def backwardQKVKernel (p : Parameters) (scale : ℝ) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let k : RegionName := "key"
  let v : RegionName := "value"
  let h : RegionName := "history"
  let «do» : RegionName := "incoming"
  let dh : RegionName := "historyGradient"
  let dq : RegionName := "queryGradient"
  let dk : RegionName := "keyGradient"
  let dv : RegionName := "valuePartials"
  with_triton_call backwardQKVSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
    b_s = tl.where(o_i[:, None] <= o_i[None, :], b_s, 0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")

        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False) * scale
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale, 0).to(b_q.dtype)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))

    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
  }

def forwardOHoistedKernel (p : Parameters) (scale : ℝ) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let k : RegionName := "key"
  let h : RegionName := "history"
  let v : RegionName := "value"
  let o : RegionName := "output"
  with_triton_call forwardOSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t, scale := scale] in triton {
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    qk_head = i_bh * s_qk_h
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + qk_head, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + qk_head, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
  }

end
def forwardH (p : Parameters) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  forwardHKernel p USE_INITIAL_STATE STORE_FINAL_STATE

def forwardO (p : Parameters) (scale : ℝ) : ComputeKernel :=
  forwardOKernel p scale

def backwardH (p : Parameters) (scale : ℝ) (USE_FINAL_GRADIENT STORE_INITIAL_GRADIENT : Bool) : ComputeKernel :=
  backwardHKernel p scale USE_FINAL_GRADIENT STORE_INITIAL_GRADIENT

def backwardQKV (p : Parameters) (scale : ℝ) : ComputeKernel :=
  backwardQKVKernel p scale

end
end VeriTile.Bench.Repairs.ChunkLinearAttn
