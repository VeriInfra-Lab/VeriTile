import bench.tritonbench_g.kldiv_compute.Proofs.Program
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.KldivCompute.LogicalExecution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- All grid programs share the logical initial arrays. This preserves the
existing allowance for coincident input/output region names and the empty grid. -/
theorem grid_inputs {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.block) (s : State α) (x y : Fin p.n → α)
    (hx : ∀ i : Fin p.n, (s.mem p.x i.val).read .real = x i)
    (hy : ∀ i : Fin p.n, (s.mem p.y i.val).read .real = y i) :
    ∃ final, Program.grid M p s = some final ∧
      (∀ i : Fin p.n, final.mem p.output i.val = Cell.mk .real (Program.scalar M (x i) (y i))) ∧
      (∀ r o, r ≠ p.output ∨ p.n ≤ o → final.mem r o = s.mem r o) := by
  obtain ⟨final, executed, memory⟩ := Program.grid_complete M p s positive
  refine ⟨final, executed, ?_, ?_⟩
  · intro i
    rw [memory]
    simp only [VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory,
      and_self, if_pos i.isLt, Program.resultCell, hx i, hy i]
    simp [VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory,
      Program.resultCell, i.isLt, hx i, hy i]
  · intro r o outside
    rw [memory]
    simp only [VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory]
    rw [if_neg (by
      rintro ⟨hr, hi⟩
      rcases outside with h | h
      · exact h hr
      · omega)]

/-- Every active access in the actual FP32 source lies in the declared
pointwise input/output extents, for any program ID and structural backend. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (hx : p.n ≤ bounds p.x) (hy : p.n ≤ bounds p.y) (ho : p.n ≤ bounds p.output) :
    TraceSafe M bounds (source p) s := by
  simp (maxSteps := 900000) [TraceSafe, Safety.traceSafeList_unfold,
    Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
    Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
    Safety.addressSafe_unfold, Safety.maskActive_unfold, Safety.safeAlg_unfold,
    source, VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel,
    VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel_fp32_fp32,
    ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, Algebra.withDefaultPrecision, resolvePrecision,
    ComputeDType.eraseDType, State.setReg, numeric, numericLt, natLt, bop, store, Region.cast]
  exact ⟨fun _ active => lt_of_lt_of_le active hx,
    fun _ active => lt_of_lt_of_le active hy, fun _ active => lt_of_lt_of_le active ho⟩

#axiomsClean grid_inputs
#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.LogicalExecution
