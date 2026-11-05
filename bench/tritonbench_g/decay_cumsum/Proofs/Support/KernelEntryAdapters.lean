import bench.tritonbench_g.decay_cumsum.Proofs.Sources.Mathematical_Kernels

/-! Repaired main/tail launches use the complete pasted Python JIT bodies in
Sources/Body.lean. The wrapper packs independent read roles with padded feature
width and supplies the actual T*width head stride. Source casts, the unmasked
preparation load, descending loop, numerical operations and final pointer
updates remain in the quotation. Invocation metadata stays outside it. -/
namespace VeriTile.Bench.Repairs.DecayCumsum
open VeriTile.Triton
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
