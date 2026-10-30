import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITRealGrid
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundSource
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety of the actual requested-format quotations. Bounds below
are independent logical windows; data-selected rereads are proved in the IO
adapter from the existing first/second selection results. -/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
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
      (JITSource.stage1 X V I (JITStage1.floatingType kind) block index64) (JITStage1.arguments s size base) := by
  cases kind <;> cases index64 <;>
    simp [TraceSafe, JITStage1.floatingType, JITSource.stage1, JITStage1.arguments,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedReal.backend, TypedReal.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      Reduction.orderedEq, TypedReal.globalLoaded, TypedReal.firstIndex,
      Option.bind, store, TileShape.allIndices, Region.cast, TypedScratch.channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
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
      (JITSource.stage2 V I O (JITStage1.floatingType kind) block) (s.setReg "mid_size" .int [] (fun _ => (size : Int))) := by
  cases kind <;>
    simp [TraceSafe, JITStage1.floatingType, JITSource.stage2,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedReal.backend, TypedReal.indexed,
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
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealSafety
