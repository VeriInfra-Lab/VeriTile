import bench.tritonbench_g.dequantize_rowwise.Kernels
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FPExecution
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- All source registers are defined by the complete straight-line kernel.
This witness is valid for arbitrary scalar primitives, input words and tails. -/
def execution {α : Type} [Inhabited α] (M : Algebra α)
    (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (u : State α) :
    {v : State α // FP.Structural.exec M (source x scale out inv n block lanes) u = some v} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, body, arguments, ComputeKernel.withScalarArguments,
    ScalarArgument.statement, ScalarArgument.fp32, ScalarArgument.integer, ComputeKernel.surfaceBody, FP.Structural.exec,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numeric, natLt, bop, ofFloat, toFloat, Region.cast, store, ComputeDType.eraseDType]
  rfl

private def beforeCode (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :=
  (source x scale out inv n block lanes).surfaceBody.take 3
private def afterCode (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :=
  (source x scale out inv n block lanes).surfaceBody.drop 4
private def rewriteSite (commuted : Bool) (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :=
  (((if commuted then indexCommutedSource else source) x scale out inv n block lanes).surfaceBody[3]?).getD
    (.effectMarker "unreachable")

private theorem original_code (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    (source x scale out inv n block lanes).surfaceBody = beforeCode x scale out inv n block lanes ++
      rewriteSite Bool.false x scale out inv n block lanes :: afterCode x scale out inv n block lanes := rfl
private theorem candidate_code (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    (indexCommutedSource x scale out inv n block lanes).surfaceBody = beforeCode x scale out inv n block lanes ++
      rewriteSite Bool.true x scale out inv n block lanes :: afterCode x scale out inv n block lanes := rfl

private theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    step M (rewriteSite Bool.false x scale out inv n block lanes) s =
      step M (rewriteSite Bool.true x scale out inv n block lanes) s := by
  simp [rewriteSite, source, indexCommutedSource, body, indexCommutedBody,
    ComputeKernel.withScalarArguments, arguments, ScalarArgument.statement,
    ComputeKernel.surfaceBody, step, run, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
  cases h : s.regs .int [] "pid" <;> simp only [h, Option.bind_none, Option.bind_some]
  congr 2
  funext i
  simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]

theorem same_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    FP.Structural.exec M (source x scale out inv n block lanes) s =
      FP.Structural.exec M (indexCommutedSource x scale out inv n block lanes) s := by
  change run M (source x scale out inv n block lanes).surfaceBody s =
    run M (indexCommutedSource x scale out inv n block lanes).surfaceBody s
  rw [original_code, candidate_code, run_append, run_append]
  apply congrArg (fun f => (run M (beforeCode x scale out inv n block lanes) s).bind f)
  funext next
  simp only [run, site_run]

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
  have same := same_run (FP.ProfiledRewrite.engine M) u x scale out inv n block lanes
  exact ⟨v.val, v.val, v.property, same.symm.trans v.property, rfl⟩

#axiomsClean equivalence

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FPExecution
