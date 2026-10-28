import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.decay_cumsum.Kernels

/-! The repair retains each original complete numerical kernel. The wrapper
packs independent read roles with a padded feature width and uses one complete
chunk launch plus a short-tail launch. Both launches use the actual packed
head stride T*width. Python program-ID and loop-bound casts to signed64 remain verbatim in the
complete source bodies below. Caller typing and source-global constants are external.
The guarded profile is FP32.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum
open VeriTile.Triton
namespace Original
noncomputable abbrev forward := VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource.forward
noncomputable abbrev prepare := VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource.prepare
noncomputable abbrev backward := VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource.backward
end Original

structure Call where
  B : Nat
  H : Nat
  T : Nat
  DK : Nat
  BT : Nat
  BK : Nat
  timeTilePositive : 0 < BT
  featureTilePositive : 0 < BK

def Call.rows (h : Call) : Nat := h.B*h.H
def Call.features (h : Call) : Nat := (h.DK+h.BK-1)/h.BK
def Call.width (h : Call) : Nat := h.features*h.BK
def Call.headStride (h : Call) : Nat := h.T*h.width
def Call.cells (h : Call) : Nat := h.rows*h.headStride

structure Phase where
  start : Nat
  count : Nat
  span : Nat

def mainPhase (h : Call) : Phase := ⟨0,h.T/h.BT,h.BT⟩
def tailPhase (h : Call) : Phase := ⟨h.T/h.BT*h.BT,1,h.T%h.BT⟩

def phases (h : Call) : List Phase :=
  (if h.T/h.BT=0 then [] else [mainPhase h]) ++
  (if h.T%h.BT=0 then [] else [tailPhase h])

inductive Kind where
  | forward | prepare | backward
  deriving DecidableEq

namespace Kernels
noncomputable section


def forwardSignature : CallSignature where
  pointers := [("g", .fp32), ("g_o", .fp32)]
  scalars := [("s_qk_h", .integer), ("s_qk_t", .integer), ("s_qk_d", .integer), ("B", .integer), ("H", .integer), ("T", .integer), ("scale", .fp32)]
  constexpr := ["BT", "BK", "DK"]
  readPorts := ["g"]
  writePorts := ["g_o"]
  sourceConstants := [("inv_ln2", "1.44269504")]

def quotedForward (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(tl.cast(BT, tl.int64)):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK
  }

def quotedForwardIndexCommuted (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call forwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (BK * i_k + tl.arange(0, BK)) < DK

    for i in range(tl.cast(BT, tl.int64)):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK
  }

def prepareSignature : CallSignature where
  pointers := [("q", .fp32), ("k", .fp32), ("g", .fp32), ("qg", .fp32), ("kg", .fp32)]
  scalars := [("s_qk_h", .integer), ("s_qk_t", .integer), ("s_qk_d", .integer), ("B", .integer), ("H", .integer), ("T", .integer), ("scale", .fp32)]
  constexpr := ["BT", "BK", "DK"]
  readPorts := ["q", "k", "g"]
  writePorts := ["qg", "kg"]

def quotedPrepare (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call prepareSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k * BK + tl.arange(0, BK))

    for i in range(tl.cast(BT, tl.int64)):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK
  }

def quotedPrepareIndexCommuted (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call prepareSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    mask = (BK * i_k + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k * BK + tl.arange(0, BK))

    for i in range(tl.cast(BT, tl.int64)):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK
  }

def backwardSignature : CallSignature where
  pointers := [("dq_inner", .fp32), ("dq_inter", .fp32), ("dk_inner", .fp32), ("dk_inter", .fp32), ("q", .fp32), ("k", .fp32), ("g", .fp32), ("dg", .fp32)]
  scalars := [("s_qk_h", .integer), ("s_qk_t", .integer), ("s_qk_d", .integer), ("B", .integer), ("H", .integer), ("T", .integer), ("scale", .fp32)]
  constexpr := ["BT", "BK", "DK"]
  readPorts := ["dq_inner", "dq_inter", "dk_inner", "dk_inter", "q", "k", "g"]
  writePorts := ["dq_inter", "dk_inter", "dg"]

def quotedBackward (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call backwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(tl.cast(BT - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.math.exp2(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq, mask=mask)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.math.exp2(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk, mask=mask)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= DK
        p_k -= DK
        p_q -= DK
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK
  }

def quotedBackwardIndexCommuted (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  with_triton_call backwardSignature call_args [s_qk_h := s_qk_h, s_qk_t := s_qk_t, s_qk_d := s_qk_d, B := B, H := H, T := T, scale := scale] in triton {
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (BK * i_k + tl.arange(0, BK)) < DK
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(tl.cast(BT - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.math.exp2(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq, mask=mask)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.math.exp2(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk, mask=mask)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= DK
        p_k -= DK
        p_q -= DK
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK
  }

def source (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : ComputeKernel :=
  match kind with
  | .forward => quotedForward "g" "go"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .prepare => quotedPrepare "q" "k" "g" "qg" "kg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .backward => quotedBackward
      "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale

/-- The existing mask-index candidate, specialized to the same packed FP32
main/tail source as `source`. Only `i_k * BK` in the lane mask is commuted. -/
def indexCommuted (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : ComputeKernel :=
  match kind with
  | .forward => quotedForwardIndexCommuted "g" "go"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .prepare => quotedPrepareIndexCommuted "q" "k" "g" "qg" "kg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .backward => quotedBackwardIndexCommuted
      "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale

end
end Kernels
end VeriTile.Bench.Repairs.DecayCumsum
