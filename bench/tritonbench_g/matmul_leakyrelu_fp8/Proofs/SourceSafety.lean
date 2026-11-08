import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.KernelBuffers
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.SourceSafety
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Grouped PID and lane initialization forms pointers without reading memory. -/
theorem prelude_safe (M : Algebra α) (D : DotAcc α) (p : Problem) (b : Buffers)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M D) bounds (prelude p b) s := by
  simp [prelude,source,TypedSource.kernel,Pasted.originalKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,State.setReg]

/-- The exact two K masks guard the actual staged A/B pointer loads. -/
theorem body_safe (M : Algebra α) (D : DotAcc α) (p : Problem) (b : Buffers)
    (initial current : State α) (k : Nat) (a : CIndex → α)
    (ctx : Context p initial current) (ha : HasAcc a current)
    (bounds : RegionName → Nat)
    (ba : ∀ i : AIndex, (k : Int)*64+i.2.1.val<(p.K : Int) → aOffset p b initial k i<bounds "A")
    (bb : ∀ i : BIndex, (k : Int)*64+i.1.val<(p.K : Int) → bOffset p b initial k i<bounds "B") :
    Safety.traceSafeList (backend M D) bounds (body p b)
      (current.setReg "k" .int [] (fun _ => (k : Int))) := by
  simp only [HasAcc] at ha
  simp [body,source,TypedSource.kernel,Pasted.originalKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
    State.setReg,ctx.memory,ctx.lk,ctx.am,ctx.bn,ha,backend]
  constructor
  · intro i j
    simpa [aOffset,safeK,ite_mul] using ba (i,j,())
  · intro i j
    simpa [bOffset,safeK,ite_mul] using bb (i,j,())

/-- Both activation branches preserve the same masked FP16 output store. -/
theorem epilogue_safe (M : Algebra α) (D : DotAcc α) (p : Problem) (b : Buffers)
    (initial current : State α) (a : CIndex → α)
    (ctx : Context p initial current) (ha : HasAcc a current)
    (bounds : RegionName → Nat)
    (bo : ∀ i : CIndex, active p initial i → cOffset p b initial i<bounds "C") :
    Safety.traceSafeList (backend M D) bounds (epilogue p b) current := by
  simp only [HasAcc] at ha
  cases he : p.enabled
  all_goals simp [epilogue,source,TypedSource.kernel,Pasted.originalKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericLt,numericEq,numericLe,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
    State.setReg,ctx.memory,ctx.pm,ctx.pn,ha,backend,he]
  all_goals
    refine ⟨?_,?_⟩
    · intro i j hi hj
      simpa [cOffset,row,col,hi,hj] using bo (i,j,()) ⟨hi,hj⟩
    · split <;> trivial

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.SourceSafety
