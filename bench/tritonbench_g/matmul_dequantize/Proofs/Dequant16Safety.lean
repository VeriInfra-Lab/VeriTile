import bench.tritonbench_g.matmul_dequantize.Sources.Mathematical_FPDequant
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatmulDequantize.Dequant16Access
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

structure Bounds (p : Parameters) (s : State A) (bounds : RegionName → Nat) : Prop where
  weight : ∀ i, active p s i → weightOffset p s i < bounds p.weight
  zero : ∀ i, active p s i → zeroOffset p s i < bounds p.zero
  scale : ∀ i, active p s i → scaleOffset p s i < bounds p.scale
  output : ∀ i, active p s i → outputOffset p s i < bounds p.output

/-- Every actually masked packed-int/scale load and typed output store in the
original quotation is in its declared region. Inactive lanes need no storage. -/
theorem source_safe (M : Algebra A) (p : Parameters) (s : State A)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) : TraceSafe M bounds (source p) s := by
  simp [TraceSafe,source,typed,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeOp_unfold,Safety.safeCompute_unfold,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,FP.Structural.eval_intShiftRight,FP.Structural.eval_intBitAnd,evalOp_unfold,
    evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    Option.bind,integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨fun i j hn hk => hb.weight (i,j,()) ⟨hk,hn⟩,
    fun i j hn hk => hb.zero (i,j,()) ⟨hk,hn⟩,
    fun i j hn hk => hb.scale (i,j,()) ⟨hk,hn⟩,
    fun i j hn hk => hb.output (i,j,()) ⟨hk,hn⟩⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatmulDequantize.Dequant16Access
