import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FPExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.DequantizeRowwise
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- All source registers are defined by the complete straight-line kernel.
This witness is valid for arbitrary scalar primitives, input words and tails. -/
def execution {α : Type} [Inhabited α] (M : Algebra α)
    (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (u : State α) :
    {v : State α // FP.Structural.exec M (source x scale out inv n block lanes) u = some v} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, TypedSource.original, ComputeKernel.surfaceBody, FP.Structural.exec,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numeric, natLt, bop, ofFloat, toFloat, Region.cast, store, ComputeDType.eraseDType]
  rfl

/-- The exact integer-index commutation preserves successful execution and
every typed memory cell for arbitrary regions and scalar reciprocal. -/
theorem equivalence (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    (⟨source x scale out inv n block lanes, []⟩ : FP.ProfiledMemory.Program)
      ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    (⟨indexCommutedSource x scale out inv n block lanes, []⟩ : FP.ProfiledMemory.Program) := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  let v := execution (FP.ProfiledRewrite.engine M) x scale out inv n block lanes u
  have same := FP.ProfiledRewrite.sound (FPProof.equivalent x scale out inv n block lanes)
    α M D hM u (by simp [FPProof.original])
  exact ⟨v.val, v.val, v.property, same.symm.trans v.property, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FPExecution
