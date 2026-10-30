import bench.tritonbench_g.mean_reduction.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MeanReduction.WholeFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

/-- Exact whole-state equality of the original and address-commuted source.
All FP32 loads, input casts, accumulator additions, reduction, integer-to-float
divisor conversion, division and stores retain their actual source order. -/
theorem tile_equal {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (s : State α) :
    exec A (typedKernel .fp32 .fp32 Bool.false X Mean M N 8 8) s =
      exec A (typedKernel .fp32 .fp32 Bool.true X Mean M N 8 8) s := by
  simp only [typedKernel, ComputeKernel.surfaceBody, FP.Structural.exec]
  rw [run,run]
  congr 1
  simp [step,evalExpr,FP.Structural.evalOp_unfold,numeric,Option.bind]
  congr 1
  funext i
  change s.pids 0*8+i.1.val = 8*s.pids 0+i.1.val
  omega

def tileState {α : Type} (pid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then pid else s.pids ax }

def runTiles {α : Type} [Inhabited α] (A : Algebra α) (v : Bool)
    (X Mean : RegionName) (M N : Nat) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (exec A (typedKernel .fp32 .fp32 v X Mean M N 8 8) (tileState pid s)).bind
      (runTiles A v X Mean M N rest)

/-- The complete launched sequence preserves all memory, registers and failure
for every opaque numerical algebra. No numerical law or dtype enumeration. -/
theorem grid_equal {α : Type} [Inhabited α] (A : Algebra α) (X Mean : RegionName)
    (M N : Nat) (tiles : List Nat) (s : State α) :
    runTiles A Bool.false X Mean M N tiles s = runTiles A Bool.true X Mean M N tiles s := by
  induction tiles generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [runTiles,tile_equal]
    cases exec A (typedKernel .fp32 .fp32 Bool.true X Mean M N 8 8) (tileState pid s) with
    | none => rfl
    | some t => exact ih t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.WholeFP
