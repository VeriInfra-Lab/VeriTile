import bench.repairs.tritonbench_g.index_select_cat.Proofs.RawKernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.IndexSelectCat.KernelSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Actual signed-index load, gathered FP32 loads and stores follow the same
active masks and pointer expressions as source execution. -/
theorem trace {α : Type} [Inhabited α] (rest : Algebra α) (p : Layout)
    (bounds : RegionName → Nat) (s : State α)
    (indices : ∀ i : Fin p.BI, s.pids 0 * p.BI + i.val < p.rows →
      s.pids 0 * p.BI + i.val < bounds p.index)
    (inputs : ∀ i, TypedGather.active p s i → TypedGather.inputOffset p s i < bounds p.source)
    (outputs : ∀ i, TypedGather.active p s i → TypedGather.outputOffset p s i < bounds p.output) :
    FP.Structural.TraceSafe (copyBackend rest) bounds (kernel p) s := by
  simp (config := { maxSteps := 500000 }) [FP.Structural.TraceSafe, kernel,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, evalDeltas, Option.bind, copyBackend, numericLt, natLt, ComputeDType.eraseDType,
    numeric, bop, ofFloat, toFloat, Region.cast, FP.Structural.store,
    State.setReg, TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  refine ⟨indices, ?_, ?_⟩
  · intro i j hi hj
    simpa [TypedGather.inputOffset, TypedGather.row, TypedGather.indexOffset,
      TypedGather.column, hi, Int.natCast_mul, Int.natCast_add] using inputs (i, j, ()) ⟨hi, hj⟩
  · intro i j hi hj
    exact outputs (i, j, ()) ⟨hi, hj⟩

#axiomsClean trace
end
end VeriTile.Bench.Repairs.IndexSelectCat.KernelSafety
