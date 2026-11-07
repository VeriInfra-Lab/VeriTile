import bench.tritonbench_g.chunk_delta_fwd.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.CallSignature

/-! Complete Python bodies, including positional block pointers and dtype
queries. Runtime strides are bound by the selected wrapper before the call. -/
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile.Triton
def callSignature : CallSignature where
  pointers := [("k",.fp32), ("v",.fp32), ("d",.fp32), ("v_new",.fp32), ("h",.fp32), ("initial_state",.fp32), ("final_state",.fp32)]
  scalars := [("s_qk_h",.integer), ("s_qk_t",.integer), ("s_qk_d",.integer), ("s_w_b",.integer), ("s_w_h",.integer), ("s_w_t",.integer), ("s_w_d",.integer), ("s_vo_h",.integer), ("s_vo_t",.integer), ("s_vo_d",.integer), ("s_h_h",.integer), ("s_h_t",.integer)]
  constexpr := ["H", "T", "K", "V", "BT", "BC", "BK", "BV", "NT", "USE_INITIAL_STATE", "STORE_FINAL_STATE"]
  readPorts := ["initial_state","k","d","v"]
  writePorts := ["h","v_new","final_state"]

def sourceKernel (p : Parameters) : ComputeKernel :=
  let k : RegionName := "key"
  let v : RegionName := "value"
  let d : RegionName := "weight"
  let v_new : RegionName := "corrected"
  let h : RegionName := "history"
  let initial_state : RegionName := "initial"
  let final_state : RegionName := "final"
  let H := p.H
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BC := p.BC
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let USE_INITIAL_STATE := p.useSeed
  let STORE_FINAL_STATE := p.storeFinal
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_w_b : Nat := p.wBatch
  let s_w_h : Nat := p.wHead
  let s_w_t : Nat := p.wTime
  let s_w_d : Nat := p.wDim
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  with_triton_call callSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_w_b := s_w_b, s_w_h := s_w_h, s_w_t := s_w_t, s_w_d := s_w_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t] in triton {
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_h_cumsum = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_d = tl.make_block_ptr(d + (i_bh // H) * s_w_b + (i_bh % H) * s_w_h, (T, K), (s_w_t, s_w_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_d = tl.load(p_d, boundary_check=(0, 1), padding_option="zero")
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
            b_v -= tl.dot(b_d, b_h.to(b_k.dtype), allow_tf32=False)
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_h_cumsum += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h += b_h_cumsum

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
  }
end VeriTile.Bench.Repairs.ChunkDeltaFwd

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
open VeriTile.Triton
abbrev Parameters := ChunkDeltaFwd.Parameters
abbrev kernel := ChunkDeltaFwd.sourceKernel
abbrev callSignature := ChunkDeltaFwd.callSignature

/-- Existing candidate: fold the K*V products in state/history addresses. -/
def addressFoldKernel (p : Parameters) : ComputeKernel :=
  let k : RegionName := "key"
  let v : RegionName := "value"
  let d : RegionName := "weight"
  let v_new : RegionName := "corrected"
  let h : RegionName := "history"
  let initial_state : RegionName := "initial"
  let final_state : RegionName := "final"
  let H := p.H
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BC := p.BC
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let USE_INITIAL_STATE := p.useSeed
  let STORE_FINAL_STATE := p.storeFinal
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_w_b : Nat := p.wBatch
  let s_w_h : Nat := p.wHead
  let s_w_t : Nat := p.wTime
  let s_w_d : Nat := p.wDim
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  with_triton_call callSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, s_w_b := s_w_b, s_w_h := s_w_h, s_w_t := s_w_t, s_w_d := s_w_d, s_vo_h := s_vo_h, s_vo_t := s_vo_t, s_vo_d := s_vo_d, s_h_h := s_h_h, s_h_t := s_h_t] in triton {
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * (K * V), (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * (K * V), (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_h_cumsum = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_d = tl.make_block_ptr(d + (i_bh // H) * s_w_b + (i_bh % H) * s_w_h, (T, K), (s_w_t, s_w_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_d = tl.load(p_d, boundary_check=(0, 1), padding_option="zero")
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
            b_v -= tl.dot(b_d, b_h.to(b_k.dtype), allow_tf32=False)
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
            b_h_cumsum += tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        b_h += b_h_cumsum

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * (K * V), (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
  }

theorem shared_source (p : Parameters) : kernel p = ChunkDeltaFwd.sourceKernel p := rfl
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
