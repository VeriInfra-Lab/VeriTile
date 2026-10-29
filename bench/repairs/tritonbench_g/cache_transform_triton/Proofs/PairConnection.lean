import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.IOResult
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.CacheTransform.PairConnection
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Exact address strength reduction, valid even when the divisor is zero.
In the admitted nonempty domain rowBytes is positive. The subtracted product
is at most the byte index, so this introduces no unsigned underflow. -/
theorem remainder (byte width : Nat) : byte % width = byte - (byte / width) * width :=
  Nat.mod_eq_sub_div_mul

theorem no_underflow (byte width : Nat) : (byte / width) * width ≤ byte :=
  Nat.div_mul_le_self byte width

/-- Complete actual source execution equality, with all registers and memory.
Only exact integer arithmetic is normalized; floating ports are never used. -/
theorem execution {A : Type} [Inhabited A] (M : Algebra A) (p : Parameters) (s : State A) :
    FP.Structural.exec M (source p) s = FP.Structural.exec M (quotientRemainder p) s := by
  simp [source,body,quotientRemainder,quotientRemainderBody,FP.Structural.exec,ComputeKernel.surfaceBody,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,ComputeDType.eraseDType,Region.cast,Option.bind,
    Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv,integralMod,IntegerExpr.sameShape,
    ←remainder]
  unfold FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,←remainder]

/-- Actual safety traces coincide, including the unconditional safe row load
and both byte masks. No source-safety or program-equality assumption is added. -/
theorem safety {A : Type} [Inhabited A] (M : Algebra A) (p : Parameters)
    (s : State A) (bounds : RegionName → Nat) :
    TraceSafe M bounds (source p) s ↔ TraceSafe M bounds (quotientRemainder p) s := by
  simp [TraceSafe,Safety.traceSafeList,Safety.stepSafe,Safety.safeAlg,Safety.safeExpr,
    Safety.safeCompute,Safety.safeOp_unfold,Safety.safeMask,Safety.safeAddress,Safety.addressSafe,
    Safety.maskActive,source,body,quotientRemainder,quotientRemainderBody,ComputeKernel.surfaceBody,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,ComputeDType.eraseDType,Region.cast,Option.bind,
    Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv,integralMod,IntegerExpr.sameShape,
    ←remainder]

#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.CacheTransform.PairConnection
