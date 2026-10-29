import bench.tritonbench_g.decay_cumsum.Kernels

/-! The repair retains each original complete numerical kernel. The wrapper
packs independent read roles with a padded feature width and uses one complete
chunk launch plus a short-tail launch. Both launches use the actual packed
head stride T*width. Python explicitly widens program IDs and loop bounds to
signed64; the natural-coordinate source below is their proved projection.
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

/-- Complete retained FP32 forward original source. -/
def quotedForward (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  triton {
    tl.region g = tl.float32, g_o = tl.float32
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_g = g + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_go = g_o + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    cum_decay = tl.zeros([$(BK)], dtype=tl.float32)
    mask = (i_k * $(BK) + tl.arange(0, $(BK)) < $(DK))
    for i in range($(BT)) {
      _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
      cum_decay += _g * $((VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource.inv_ln2 : ℝ))
      tl.store(p_go, (cum_decay).to(tl.float32), mask=mask)
      p_g += $(DK)
      p_go += $(DK)
    }
  }

/-- Complete retained FP32 forward candidate source. -/
def quotedForwardIndexCommuted (g g_o : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  triton {
    tl.region g = tl.float32, g_o = tl.float32
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_g = g + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_go = g_o + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    cum_decay = tl.zeros([$(BK)], dtype=tl.float32)
    mask = ($(BK) * i_k + tl.arange(0, $(BK)) < $(DK))
    for i in range($(BT)) {
      _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
      cum_decay += _g * $((VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource.inv_ln2 : ℝ))
      tl.store(p_go, (cum_decay).to(tl.float32), mask=mask)
      p_g += $(DK)
      p_go += $(DK)
    }
  }

/-- Complete retained FP32 prepare original source. -/
def quotedPrepare (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[q, k, g], outputs=[qg, kg]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, qg = tl.float32, kg = tl.float32
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_g = g + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_k = k + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_qg = qg + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_kg = kg + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    mask = (i_k * $(BK) + tl.arange(0, $(BK)) < $(DK))
    last_decay = tl.load(g + i_bh * $(s_qk_h) + (i_c * $(BT) + $(BT) - $(1)) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK)))
    for i in range($(BT)) {
      _q = tl.load(p_q, mask=mask, other=0)
      _k = tl.load(p_k, mask=mask, other=0)
      _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
      _q *= tl.math.exp2(_g) * $((scale : ℝ))
      _k *= tl.math.exp2(last_decay - _g)
      tl.store(p_kg, (_k).to(tl.float32), mask=mask)
      tl.store(p_qg, (_q).to(tl.float32), mask=mask)
      p_q += $(DK)
      p_g += $(DK)
      p_k += $(DK)
      p_kg += $(DK)
      p_qg += $(DK)
    }
  }

/-- Complete retained FP32 prepare candidate source. -/
def quotedPrepareIndexCommuted (q k g qg kg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[q, k, g], outputs=[qg, kg]
    tl.region q = tl.float32, k = tl.float32, g = tl.float32, qg = tl.float32, kg = tl.float32
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_g = g + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_k = k + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_qg = qg + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    p_kg = kg + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK))
    mask = ($(BK) * i_k + tl.arange(0, $(BK)) < $(DK))
    last_decay = tl.load(g + i_bh * $(s_qk_h) + (i_c * $(BT) + $(BT) - $(1)) * $(DK) + i_k * $(BK) + tl.arange(0, $(BK)))
    for i in range($(BT)) {
      _q = tl.load(p_q, mask=mask, other=0)
      _k = tl.load(p_k, mask=mask, other=0)
      _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
      _q *= tl.math.exp2(_g) * $((scale : ℝ))
      _k *= tl.math.exp2(last_decay - _g)
      tl.store(p_kg, (_k).to(tl.float32), mask=mask)
      tl.store(p_qg, (_q).to(tl.float32), mask=mask)
      p_q += $(DK)
      p_g += $(DK)
      p_k += $(DK)
      p_kg += $(DK)
      p_qg += $(DK)
    }
  }

/-- Complete retained FP32 backward original source. -/
def quotedBackward (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[dq_inner, dq_inter, dk_inner, dk_inter, q, k, g], outputs=[dq_inter, dk_inter, dg]
    tl.region dq_inner = tl.float32, dq_inter = tl.float32, dk_inner = tl.float32, dk_inter = tl.float32, q = tl.float32, k = tl.float32, g = tl.float32, dg = tl.float32
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_g = g + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dg = dg + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dq_inner = dq_inner + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dk_inner = dk_inner + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dq_inter = dq_inter + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dk_inter = dk_inter + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
    mask = (i_k * $(BK) + tl.arange(0, $(BK)) < $(DK))
    last_g = tl.zeros([$(BK)], dtype=tl.float32)
    for j in range($(BT) - $(1), -$(1), -$(1)) {
      _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
      if (j == $(BT) - $(1)) {
        last_g = _g
      }
      _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
      _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
      _dq2 *= tl.math.exp2(_g)
      _dq = _dq1 + _dq2
      tl.store(p_dq_inter, (_dq).to(tl.float32), mask=mask)
      _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
      _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
      _dk2 *= tl.math.exp2(last_g - _g)
      _dk = _dk1 + _dk2
      tl.store(p_dk_inter, (_dk).to(tl.float32), mask=mask)
      _q = tl.load(p_q, mask=mask, other=0)
      _k = tl.load(p_k, mask=mask, other=0)
      _dg = _dq * _q - _dk * _k
      cum_grad_dg += _dg
      tl.store(p_dg, (cum_grad_dg).to(tl.float32), mask=mask)
      p_g -= $(DK)
      p_k -= $(DK)
      p_q -= $(DK)
      p_dq_inner -= $(DK)
      p_dk_inner -= $(DK)
      p_dq_inter -= $(DK)
      p_dk_inter -= $(DK)
      p_dg -= $(DK)
    }
  }

/-- Complete retained FP32 backward candidate source. -/
def quotedBackwardIndexCommuted (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (s_qk_h s_qk_t s_qk_d B H T BT BK DK : Nat) (scale : ℝ) : ComputeKernel :=
  triton {
    tl.ports inputs=[dq_inner, dq_inter, dk_inner, dk_inter, q, k, g], outputs=[dq_inter, dk_inter, dg]
    tl.region dq_inner = tl.float32, dq_inter = tl.float32, dk_inner = tl.float32, dk_inter = tl.float32, q = tl.float32, k = tl.float32, g = tl.float32, dg = tl.float32
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_g = g + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dg = dg + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dq_inner = dq_inner + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dk_inner = dk_inner + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dq_inter = dq_inter + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    p_dk_inter = dk_inter + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
    cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
    mask = ($(BK) * i_k + tl.arange(0, $(BK)) < $(DK))
    last_g = tl.zeros([$(BK)], dtype=tl.float32)
    for j in range($(BT) - $(1), -$(1), -$(1)) {
      _g = (tl.load(p_g, mask=mask, other=0)).to(tl.float32)
      if (j == $(BT) - $(1)) {
        last_g = _g
      }
      _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
      _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
      _dq2 *= tl.math.exp2(_g)
      _dq = _dq1 + _dq2
      tl.store(p_dq_inter, (_dq).to(tl.float32), mask=mask)
      _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
      _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
      _dk2 *= tl.math.exp2(last_g - _g)
      _dk = _dk1 + _dk2
      tl.store(p_dk_inter, (_dk).to(tl.float32), mask=mask)
      _q = tl.load(p_q, mask=mask, other=0)
      _k = tl.load(p_k, mask=mask, other=0)
      _dg = _dq * _q - _dk * _k
      cum_grad_dg += _dg
      tl.store(p_dg, (cum_grad_dg).to(tl.float32), mask=mask)
      p_g -= $(DK)
      p_k -= $(DK)
      p_q -= $(DK)
      p_dq_inner -= $(DK)
      p_dk_inner -= $(DK)
      p_dq_inter -= $(DK)
      p_dk_inter -= $(DK)
      p_dg -= $(DK)
    }
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
