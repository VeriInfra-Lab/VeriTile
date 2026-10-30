import bench.repairs.tritonbench_g.masked_select.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MaskedSelect.AccessSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect TypedCopy Whole StableSelection
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- All three masked reads and the actual prefix-addressed scatter are safe.
The integer prefix/selection facts are logical inputs, not numerical laws. -/
theorem source_safe {α : Type} [Inhabited α] (rest : Algebra α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i, i < p.size → i < bounds p.input)
    (selection : ∀ i, i < p.size → i < bounds p.selection.cast)
    (cumulative : ∀ i, i < p.size → i < bounds p.prefixSum.cast)
    (output : ∀ j : TileIndex [p.width], TypedCopy.active p s j →
      outputOffset p s j < bounds p.output) :
    FP.Structural.TraceSafe (copyBackend rest) bounds (kernel p) s := by
  simp [FP.Structural.TraceSafe, kernel, Whole.kernel, TypedCopy.source,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, store, Option.bind,
    ComputeDType.eraseDType, numeric, numericLt, bop, natLt,
    ofFloat, toFloat, Region.cast, State.setReg, copyBackend]
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro j hj; exact input _ hj
  · intro j hj; exact selection _ hj
  · intro j hj; exact cumulative _ hj
  · intro j hs hn _
    simpa [outputOffset, inputOffset, hs] using output (j, PUnit.unit) ⟨hn, hs⟩

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.AccessSafety
