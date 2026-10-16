import bench.tritonbench_g.fused_recurrent_retention.Proofs.Program
import VeriTile.Triton.Float.Structural

/-! Complete typed forward and backward executions. The candidate changes
only a discrete natural-number comparison; no floating identity is assumed.
This equality preserves failures and is separate from successful-execution
witnesses and from the native backend refinement. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.FPProof
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def keyOffset (p : Layout) : Op .nat [p.BK] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "i_k") (.constNat p.BK))
    (.arange p.BK)

def keyMask (p : Layout) (swapped : Bool) : ComputeStmt :=
  .assign .bool [p.BK] "mask_bk" (.alg (if swapped then
    .gt .nat .scalarL (.constNat p.DK) (keyOffset p)
    else .lt .nat .scalarR (keyOffset p) (.constNat p.DK)))

theorem key_mask (p : Layout) {α : Type} [Inhabited α]
    (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.run A [keyMask p Bool.false] u =
      FP.Structural.run A [keyMask p Bool.true] u := by
  cases hi : u.regs .nat [] "i_k" <;>
    simp [keyMask,keyOffset,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalOp_unfold,FP.Structural.numericLt,
    FP.Structural.natLt,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind,bind,hi]
  apply congrArg (u.setReg "mask_bk" .bool [p.BK])
  funext i
  rfl

theorem forward_split (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    (Program.forward swapped dtype out_dtype p r hasSeed storeFinal).surfaceBody =
      (Program.forward Bool.false dtype out_dtype p r hasSeed storeFinal).surfaceBody.take 9 ++
      [keyMask p swapped] ++
      (Program.forward Bool.false dtype out_dtype p r hasSeed storeFinal).surfaceBody.drop 10 := by
  cases swapped <;> rfl

theorem backward_split (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) :
    (Program.backward swapped dtype out_dtype p r hasSeed).surfaceBody =
      (Program.backward Bool.false dtype out_dtype p r hasSeed).surfaceBody.take 10 ++
      [keyMask p swapped] ++
      (Program.backward Bool.false dtype out_dtype p r hasSeed).surfaceBody.drop 11 := by
  cases swapped <;> rfl

theorem forward_whole_state (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.exec A (Program.forward Bool.false dtype out_dtype p r hasSeed storeFinal) u =
      FP.Structural.exec A (Program.forward Bool.true dtype out_dtype p r hasSeed storeFinal) u := by
  change FP.Structural.run A
      (Program.forward Bool.false dtype out_dtype p r hasSeed storeFinal).surfaceBody u =
    FP.Structural.run A
      (Program.forward Bool.true dtype out_dtype p r hasSeed storeFinal).surfaceBody u
  rw [forward_split Bool.false,forward_split Bool.true]
  simp only [FP.Structural.run_append,key_mask]

theorem backward_whole_state (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.exec A (Program.backward Bool.false dtype out_dtype p r hasSeed) u =
      FP.Structural.exec A (Program.backward Bool.true dtype out_dtype p r hasSeed) u := by
  change FP.Structural.run A
      (Program.backward Bool.false dtype out_dtype p r hasSeed).surfaceBody u =
    FP.Structural.run A
      (Program.backward Bool.true dtype out_dtype p r hasSeed).surfaceBody u
  rw [backward_split Bool.false,backward_split Bool.true]
  simp only [FP.Structural.run_append,key_mask]

#axiomsClean key_mask
#axiomsClean forward_split
#axiomsClean backward_split
#axiomsClean forward_whole_state
#axiomsClean backward_whole_state
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.FPProof
