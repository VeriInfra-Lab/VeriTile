import bench.tritonbench_g.triton_argmax.Proofs.JITInteger
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
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
      (JITSource.stage1 X V I (JITStage1.floatingType kind) block index64) (JITStage1.arguments s size base) := by
  cases kind <;> cases index64 <;>
    simp [TraceSafe, JITStage1.floatingType, JITSource.stage1, JITStage1.arguments,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedScratch.backend, Typed.backend, TypedScratch.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      TypedReduction.self_equal, TypedReduction.presence, TypedReduction.selected,
      flag, clean, Typed.globalLoaded, Typed.firstIndex, TypedScratch.Rounding.cast,
      Option.bind, store, TileShape.allIndices, Region.cast, TypedScratch.channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals try simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast]
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
    TraceSafe (JITInteger.backend kind primitives rest) bounds
      (JITSource.stage1 X V I (JITInteger.sourceType kind) block index64) (JITStage1.arguments s size base) := by
  cases kind <;> cases index64 <;>
    simp [TraceSafe, JITInteger.sourceType, JITSource.stage1, JITStage1.arguments,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      JITInteger.backend, Typed.backend,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      TypedReduction.self_equal, TypedReduction.presence, TypedReduction.selected,
      flag, clean, Typed.globalLoaded, Typed.firstIndex, collectValues?,
      Option.bind, store, TileShape.allIndices, Region.cast, collectValues?, -Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul,
      Formats.IntegerKind.lower,Formats.IntegerKind.upper,Formats.IntegerKind.exponent,
      Formats.IntegerKind.gap,Formats.IntegerKind.signed,Formats.IntegerKind.bits,ScalarBinding.lowerBits,ScalarBinding.upperBits]
  all_goals try simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast]
  all_goals norm_num
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
      (JITSource.stage2 V I O (JITStage1.floatingType kind) block) (s.setReg "mid_size" .int [] (fun _ => (size : Int))) := by
  have pad16 : round.fp16 (4286578688#32) = (4286578688#32) := round.fp16Padding
  have padbf : round.bf16 (4286578688#32) = (4286578688#32) := round.bf16Padding
  cases kind <;>
    simp [TraceSafe, JITStage1.floatingType, JITSource.stage2,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      TypedScratch.backend, Typed.backend, TypedScratch.indexed,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
      TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
      natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax,
      TypedReduction.self_equal, TypedReduction.presence, TypedReduction.selected,
      flag, clean, TypedScratch.secondIndex, TypedScratch.channel,
      Option.bind, store, TileShape.allIndices, Region.cast, pad16, padbf,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals try simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast]
  all_goals
    refine ⟨?_, ?_, output⟩
    · intro i active
      exact values _ active
    · simpa [TypedScratch.secondIndex, TypedScratch.channel, TypedReduction.selected,
        TypedReduction.presence, flag, clean, ofFloat, toFloat] using selected

theorem integer_second (kind : Formats.IntegerKind) (rest : Algebra Word) (s : State Word)
    (V : RegionName) (I O : Region .int) (size block : Nat) (bounds : RegionName → Nat)
    (values : ∀ i, i < size → i < bounds V)
    (selected : JITInteger.secondIndex kind s V size block < bounds (Region.cast I))
    (output : 0 < bounds (Region.cast O)) :
    TraceSafe rest bounds (JITSource.stage2 V I O (JITInteger.sourceType kind) block)
      (s.setReg "mid_size" .int [] (fun _ => (size : Int))) := by
  cases kind <;>
    simp [TraceSafe, JITInteger.sourceType, JITSource.stage2,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold,
      TileShape.axisDim, TileShape.insertAxisIndex, TileShape.eraseAxis,
      ofFloat, toFloat, bop, numeric, numericLt, numericEq, natLt,
      TileShape.discreteIntArgMax, JITInteger.secondIndex, JITInteger.read,
      Option.bind, store, TileShape.allIndices, Region.cast, Op.boolToInt,
      Formats.IntegerKind.padding,Formats.IntegerKind.lower,Formats.IntegerKind.exponent,
      Formats.IntegerKind.signed,Formats.IntegerKind.bits] at selected ⊢
  all_goals refine ⟨fun _ hi => values _ hi, ?_, output⟩
  all_goals try exact selected
  have lanes : (fun lane : Fin block =>
      if lane.val < size ∧ (s.mem V lane.val).read .bool = Bool.true then (1 : Int) else 0) =
      (fun lane : Fin block => if lane.val < size then
        (if (s.mem V lane.val).read .bool then (1 : Int) else 0) else 0) := by
    funext lane
    by_cases h : lane.val < size <;> simp [h]
  intro _
  exact (congrArg (fun f => TileShape.leftmostIntArgMax f < bounds (Region.cast I)) lanes).mpr selected

#axiomsClean native_first
#axiomsClean integer_first
#axiomsClean native_second
#axiomsClean integer_second
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITSafety
