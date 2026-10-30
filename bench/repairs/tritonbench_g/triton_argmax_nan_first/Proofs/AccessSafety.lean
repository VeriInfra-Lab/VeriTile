import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealGrid
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundSource
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety of the actual requested-format quotations. Bounds below
are independent logical windows; data-selected rereads are proved in the IO
adapter from the existing first/second selection results. -/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.AccessSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem real_first (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (s : State Math.Value) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0 < block)
    (bounds : RegionName → Nat)
    (input : ∀ i, i < size → i < bounds X)
    (selected : TypedReal.firstIndex s X size block base positive < bounds X)
    (value : s.pids 0 + base < bounds V)
    (index : s.pids 0 + base < bounds (Region.cast I)) :
    TraceSafe (TypedReal.backend rest) bounds
      (TypedScratch.firstKernel kind X V I size block base index64) s := by
  cases kind <;> cases index64 <;>
    simp [TraceSafe, TypedScratch.firstKernel, TypedSource.stage1FP32,
      TypedSource.stage1FP16, TypedSource.stage1BF16, TypedSource.stage1FP64,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedReal.backend, State.setReg, TypedReal.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      Reduction.orderedEq, TypedReal.globalLoaded, TypedReal.firstIndex,
      Option.bind, store, TileShape.allIndices, Region.cast, TypedScratch.channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals try simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast]
  all_goals
    refine ⟨?_, ?_, value, index⟩
    · intro i active
      exact input _ active
    · exact selected

theorem real_second (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (s : State Math.Value) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0 < block) (bounds : RegionName → Nat)
    (values : ∀ i, i < size → i < bounds V)
    (selected : TypedReal.secondIndex kind s V size block positive < bounds (Region.cast I))
    (output : 0 < bounds (Region.cast O)) :
    TraceSafe (TypedReal.backend rest) bounds
      (TypedScratch.secondKernel kind V I O size block) s := by
  cases kind <;>
    simp [TraceSafe, TypedScratch.secondKernel, TypedSource.stage2FP32,
      TypedSource.stage2FP16, TypedSource.stage2BF16, TypedSource.stage2FP64,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedReal.backend, State.setReg, TypedReal.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      Reduction.orderedEq, TypedReal.secondIndex,
      Option.bind, store, TileShape.allIndices, Region.cast, TypedScratch.channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals
    refine ⟨?_, selected, output⟩
    intro i active
    exact values _ active

#axiomsClean real_first
#axiomsClean real_second
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.AccessSafety
