import bench.tritonbench_g.matmul_tma.Proofs.SourceSafety
import bench.tritonbench_g.matmul_tma.Proofs.Sources.Native
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourcePair
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section
local instance : Inhabited StaticChecked.Native.Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The direct-store candidate succeeds with the same full typed memory.
Private c-register formats differ in the FP16 branch, so whole states are
not identified. The selected native cast profile fixes both precision contexts
to the same conversion instruction, with an identity same-format store. -/
theorem candidate_memory (rest : Algebra StaticChecked.Native.Word)
    (I : StaticChecked.Native.Instructions) (D : Dot StaticChecked.Native.Word)
    (p : Parameters) (half : Bool) (s : State StaticChecked.Native.Word) :
    let M := StaticChecked.Native.algebra rest I;
    ∃ t, FP.Structural.exec (backend M D) (sourceDirectOutput p half) s = some t ∧
      t.mem = (scatter p half s (outputValues M D p half s)).mem := by
  cases half <;>
    simp [sourceDirectOutput, directOutputTypedKernel, FP.Structural.exec,
      ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      store, Option.bind, ComputeDType.eraseDType, numeric, bop, Region.cast,
      BlockPtr.inBounds, BlockPtr.address_2d_zero_offsets, TileShape.indexToList,
      backend, outputValues, outputValue, aValues, bValues, aOffset, bOffset,
      cOffset, outputDType, scatter, ofFloat, toFloat, StaticChecked.Native.algebra]
  · exact scatter_memory_congr (half := Bool.false) _ _ _ _ rfl
  · exact scatter_memory_congr (half := Bool.true) _ _ _ _ rfl

theorem source_exec (rest : Algebra StaticChecked.Native.Word)
    (I : StaticChecked.Native.Instructions) (D : Dot StaticChecked.Native.Word)
    (p : Parameters) (half : Bool) (s t : State StaticChecked.Native.Word)
    (ran : exec (backend (StaticChecked.Native.algebra rest I) D) (source p half) s = some t) :
    ∃ u, exec (backend (StaticChecked.Native.algebra rest I) D) (sourceDirectOutput p half) s = some u ∧ u.mem = t.mem := by
  obtain ⟨u,hu,hm⟩ := candidate_memory rest I D p half s
  obtain ⟨v,hv,hvm⟩ := execution_memory (StaticChecked.Native.algebra rest I) D p half s
  have eq := Option.some.inj (hv.symm.trans ran)
  subst t
  exact ⟨u,hu,hm.trans hvm.symm⟩

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (bounds : RegionName → Nat) :
    TraceSafe (backend M D) bounds (source p half) s ↔
      TraceSafe (backend M D) bounds (sourceDirectOutput p half) s := by
  cases half
  all_goals simp [TraceSafe, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, Safety.safeAlg_unfold, source, typedKernel,sourceDirectOutput,directOutputTypedKernel,
      FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, Region.cast, BlockPtr.inBounds, BlockPtr.address_2d_zero_offsets,
      TileShape.indexToList, backend, outputValues, aValues, bValues,
      aOffset, bOffset, cOffset, outputDType, ofFloat, toFloat]
#axiomsClean candidate_memory
#axiomsClean source_exec
#axiomsClean source_safe
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourcePair
