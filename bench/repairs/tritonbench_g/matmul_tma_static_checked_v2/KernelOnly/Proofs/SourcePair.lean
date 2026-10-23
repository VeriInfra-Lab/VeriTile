import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.SourceSafety
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourcePair
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section
local instance : Inhabited StaticChecked.Native.Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The complete source transformation uses the selected native model's
same-format identity only; its dot instructions and FP16 conversion are intact. -/
theorem source_exec (rest : Algebra StaticChecked.Native.Word)
    (I : StaticChecked.Native.Instructions) (profile : StaticChecked.Native.Profile)
    (p : Parameters) (half : Bool) (s : State StaticChecked.Native.Word) :
    exec (backend (StaticChecked.Native.algebra rest I) (StaticChecked.Native.sourceDot I profile)) (source p half) s =
      exec (backend (StaticChecked.Native.algebra rest I) (StaticChecked.Native.sourceDot I profile)) (sourceDirectOutput p half) s := by
  cases half
  · simp [sourceDirectOutput,source,typed,outputDType,FP.Structural.exec,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,State.setReg,
      backend,StaticChecked.Native.algebra,StaticChecked.Native.sourceDot,ofFloat,toFloat,ComputeDType.eraseDType]
  · rfl

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (bounds : RegionName → Nat) :
    TraceSafe (backend M D) bounds (source p half) s ↔
      TraceSafe (backend M D) bounds (sourceDirectOutput p half) s := by
  cases half
  · simp [TraceSafe, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, Safety.safeAlg_unfold, source, typed,sourceDirectOutput,
      FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, Region.cast, BlockPtr.inBounds, BlockPtr.address_2d_zero_offsets,
      TileShape.indexToList, backend, outputValues, aValues, bValues,
      aOffset, bOffset, cOffset, outputDType, ofFloat, toFloat]
  · rfl
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourcePair
