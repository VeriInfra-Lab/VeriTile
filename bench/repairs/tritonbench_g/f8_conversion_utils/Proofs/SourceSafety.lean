import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Source
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

private theorem trace_last {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionBounds) (stmt : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [stmt] s ↔ Safety.stepSafe M bounds stmt s := by
  rw [Safety.traceSafeList_cons]
  simp

/-- Every actual typed load/store has a backed active address. The two decoder
stores are both checked against the same output window. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B : Nat) (s : State α) (bounds : RegionBounds)
    (input : ∀ j : Fin B, s.pids 0*B+j.val<N → s.pids 0*B+j.val<bounds X)
    (output : ∀ j : Fin B, s.pids 0*B+j.val<N → s.pids 0*B+j.val<bounds Y) :
    TraceSafe M bounds (kernel p Y X N B) s := by
  cases Y
  cases X
  cases p
  all_goals
    simp +contextual (maxSteps := 500000) [trace_last,TraceSafe,kernel,toHalf,toFP8,
      sourceDecodeE5,sourceDecodeE4,sourceEncodeHalfE5,sourceEncodeHalfE4,
      sourceEncodeSingleE5,sourceEncodeSingleE4,
      VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource.toHalf,
      VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource.toFP8,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,numeric,natLt,bop,
      FP.Structural.store,Region.cast,ofFloat,toFloat,State.setReg,Option.bind,
      VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_regs,
      VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_mem,input,output]

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source
