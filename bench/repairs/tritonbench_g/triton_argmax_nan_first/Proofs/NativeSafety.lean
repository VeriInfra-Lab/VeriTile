import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundSource
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.AccessSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem native_first (kind : TypedScratch.Kind) (round : TypedScratch.Rounding)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0 < block)
    (bounds : RegionName → Nat)
    (input : ∀ i, i < size → i < bounds X)
    (selected : Typed.firstIndex s X size block base positive < bounds X)
    (value : s.pids 0 + base < bounds V)
    (index : s.pids 0 + base < bounds (Region.cast I)) :
    TraceSafe (TypedScratch.backend round rest) bounds
      (TypedScratch.firstKernel kind X V I size block base index64) s := by
  cases kind <;> cases index64 <;>
    simp [TraceSafe, TypedScratch.firstKernel, TypedSource.stage1FP32,
      TypedSource.stage1FP16, TypedSource.stage1BF16, TypedSource.stage1FP64,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedScratch.backend, Typed.backend, State.setReg, TypedScratch.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      TypedReduction.self_equal, TypedReduction.presence, TypedReduction.selected,
      flag, clean, Typed.globalLoaded, Typed.firstIndex, TypedScratch.Rounding.cast,
      Option.bind, store, TileShape.allIndices, Region.cast, TypedScratch.channel]
  all_goals
    refine ⟨?_, ?_, value, index⟩
    · intro i active
      exact input _ active
    · simpa [Typed.firstIndex, Typed.globalLoaded, TypedReduction.selected,
        TypedReduction.presence, flag, clean] using selected

theorem integer_first (kind : Formats.IntegerKind) (primitives : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0 < block)
    (bounds : RegionName → Nat)
    (input : ∀ i, i < size → i < bounds X)
    (selected : Typed.firstIndex s X size block base positive < bounds X)
    (value : s.pids 0 + base < bounds V)
    (index : s.pids 0 + base < bounds (Region.cast I)) :
    TraceSafe (TypedIntegerMacro.backend kind primitives rest) bounds
      (TypedIntegerMacro.kernel X V I size block base index64) s := by
  cases index64 <;>
    simp [TraceSafe, TypedIntegerMacro.kernel, TypedSource.stage1Integer,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedIntegerMacro.backend, Typed.backend, State.setReg,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      TypedReduction.self_equal, TypedReduction.presence, TypedReduction.selected,
      flag, clean, Typed.globalLoaded, Typed.firstIndex, collectValues?,
      Option.bind, store, TileShape.allIndices, Region.cast]
  all_goals
    refine ⟨?_, ?_, value, index⟩
    · intro i active
      exact input _ active
    · simpa [Typed.firstIndex, Typed.globalLoaded, TypedReduction.selected,
        TypedReduction.presence, flag, clean] using selected

theorem native_second (kind : TypedScratch.Kind) (round : TypedScratch.Rounding)
    (rest : Algebra Word) (s : State Word) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0 < block) (bounds : RegionName → Nat)
    (values : ∀ i, i < size → i < bounds V)
    (selected : TypedScratch.secondIndex kind s V size block positive < bounds (Region.cast I))
    (output : 0 < bounds (Region.cast O)) :
    TraceSafe (TypedScratch.backend round rest) bounds
      (TypedScratch.secondKernel kind V I O size block) s := by
  have pad16 : round.fp16 (4286578688#32) = (4286578688#32) := round.fp16Padding
  have padbf : round.bf16 (4286578688#32) = (4286578688#32) := round.bf16Padding
  cases kind <;>
    simp [TraceSafe, TypedScratch.secondKernel, TypedSource.stage2FP32,
      TypedSource.stage2FP16, TypedSource.stage2BF16, TypedSource.stage2FP64,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedScratch.backend, Typed.backend, State.setReg, TypedScratch.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      TypedReduction.self_equal, TypedReduction.presence, TypedReduction.selected,
      flag, clean, TypedScratch.secondIndex, TypedScratch.channel,
      Option.bind, store, TileShape.allIndices, Region.cast, pad16, padbf]
  all_goals
    refine ⟨?_, ?_, output⟩
    · intro i active
      exact values _ active
    · simpa [TypedScratch.secondIndex, TypedScratch.channel, TypedReduction.selected,
        TypedReduction.presence, flag, clean, ofFloat, toFloat] using selected

theorem integer_second (rest : Algebra Word) (s : State Word) (V I O : Region .int)
    (size block : Nat) (padding : Int) (bounds : RegionName → Nat)
    (values : ∀ i, i < size → i < bounds (Region.cast V))
    (selected : TypedInteger.selected s V size block padding < bounds (Region.cast I))
    (output : 0 < bounds (Region.cast O)) :
    TraceSafe rest bounds (TypedInteger.stage2 V I O size block padding) s := by
  simp [TraceSafe, TypedInteger.stage2, TypedSource.stage2Integer,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
    Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
    Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
    Safety.addressSafe_unfold, Safety.maskActive_unfold,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, State.setReg,
    TileShape.axisDim, TileShape.insertAxisIndex, TileShape.eraseAxis,
    ofFloat, toFloat, bop, numeric, numericLt, numericEq, natLt,
    TileShape.discreteIntArgMax, TypedInteger.selected,
    Option.bind, store, TileShape.allIndices, Region.cast]
  exact ⟨fun _ hi => values _ hi, selected, output⟩

#axiomsClean native_first
#axiomsClean integer_first
#axiomsClean native_second
#axiomsClean integer_second
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.AccessSafety
