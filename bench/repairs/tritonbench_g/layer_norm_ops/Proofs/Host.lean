import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardWhole
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Frame
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridHost
namespace VeriTile.Bench.Repairs.LayerNormOps.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
namespace Generic
export VeriTile.Bench.Repairs.LayerNormLiger.Host (Tensor Scalar Matrix axisSum)
end Generic
namespace Grid
export VeriTile.Bench.Repairs.LayerNormLiger.Grid (run reference complete)
end Grid
noncomputable section

abbrev Tensor := Generic.Tensor
abbrev Scalar := Generic.Scalar
abbrev Matrix := Generic.Matrix

def matrix (r : RegionName) (rows columns : Nat) : Tensor := ⟨r,[rows,columns],.fp32⟩
def vector (r : RegionName) (size : Nat) : Tensor := ⟨r,[size],.fp32⟩

structure ForwardReturn (α : Type) where
  state : State α
  y : Tensor
  mean : Option Tensor
  rstd : Tensor
  z : Tensor

def forwardFinish {α : Type} (p : ForwardParameters) (rows : Nat) (s : State α) : ForwardReturn α where
  state := s
  y := matrix p.y rows p.columns
  mean := if p.rms then none else some (vector p.mean rows)
  rstd := vector p.rstd rows
  z := matrix (if p.storeResidual then p.residualOut else p.x) rows p.columns

def forward {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (rows : Nat) (s : State α) : Option (ForwardReturn α) := do
  let t ← Grid.run M (forwardKernel p) s rows
  pure (forwardFinish p rows t)

theorem forward_complete {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (rows : Nat) (s : State α) :
    forward M p rows s = some (forwardFinish p rows (Grid.reference (Forward.reference M p) s rows)) := by
  rw [forward, Grid.complete M (forwardKernel p) (Forward.reference M p) (Forward.complete_run M p)]
  rfl

structure BackwardReturn (α : Type) where
  state : State α
  dx : Tensor
  dw : Tensor
  db : Option Tensor
  dResidual : Option Tensor
  recomputedY : Option Tensor

def partialMatrix (p : BackwardParameters) (r : RegionName) (programs : Nat) : Matrix :=
  ⟨r,programs,p.columns,p.columns⟩

def backwardFinish {α : Type} [Inhabited α] (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName)
    (hasResidual : Bool) (s : State α) : BackwardReturn α where
  state := let t := Generic.axisSum ops (partialMatrix p p.dw programs) pw dw s
    if p.hasBias then Generic.axisSum ops (partialMatrix p p.db programs) pb db t else t
  dx := matrix p.dx p.rows p.columns
  dw := vector dw p.columns
  db := if p.hasBias then some (vector db p.columns) else none
  dResidual := if hasResidual then
    some (matrix (if p.storeDres then p.dresIn else p.dx) p.rows p.columns) else none
  recomputedY := if p.recompute then some (matrix p.y p.rows p.columns) else none

def backward {α : Type} [Inhabited α] (M : Algebra α) (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName)
    (hasResidual : Bool) (s : State α) : Option (BackwardReturn α) := do
  let t ← Grid.run M (backwardKernel p) s programs
  pure (backwardFinish ops p programs pw pb dw db hasResidual t)

theorem backward_complete {α : Type} [Inhabited α] (M : Algebra α) (ops : Scalar α)
    (p : BackwardParameters) (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (hasResidual : Bool) (s : State α) :
    backward M ops p programs pw pb dw db hasResidual s =
      some (backwardFinish ops p programs pw pb dw db hasResidual
        (Grid.reference (Backward.reference M p) s programs)) := by
  rw [backward, Grid.complete M (backwardKernel p) (Backward.reference M p) (Backward.complete_run M p)]
  rfl

/-- The backward call consumes the exact state produced by forward. The
ordinary descriptor contract below binds its X to returned Z and its saved
statistics to those outputs; it assumes no numerical statistic equation. -/
def composed {α : Type} [Inhabited α] (M : Algebra α) (ops : Scalar α)
    (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State α) :
    Option (ForwardReturn α × BackwardReturn α) := do
  let f ← forward M fp rows s
  let b ← backward M ops bp programs pw pb dw db fp.hasResidual f.state
  pure (f,b)

theorem composed_complete {α : Type} [Inhabited α] (M : Algebra α) (ops : Scalar α)
    (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State α) :
    composed M ops fp bp rows programs pw pb dw db s =
      let f := forwardFinish fp rows (Grid.reference (Forward.reference M fp) s rows)
      some (f,backwardFinish ops bp programs pw pb dw db fp.hasResidual
        (Grid.reference (Backward.reference M bp) f.state programs)) := by
  simp [composed, forward_complete, backward_complete, Option.bind]

theorem returned_residual_alias {α : Type} [Inhabited α] (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State α)
    (h : p.storeDres = Bool.false) :
    (backwardFinish ops p programs pw pb dw db Bool.true s).dResidual =
      some (backwardFinish ops p programs pw pb dw db Bool.true s).dx := by
  simp [backwardFinish,h]

#axiomsClean forward_complete
#axiomsClean backward_complete
#axiomsClean composed_complete
#axiomsClean returned_residual_alias
end
end VeriTile.Bench.Repairs.LayerNormOps.Host
