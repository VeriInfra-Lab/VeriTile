import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawMemory
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSafety
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prelude_safe (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M D) bounds (RawSource.prelude p st) s := by
  cases st <;> simp [RawSource.prelude,Source.typed,Source.halfSource,Source.byteSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,dtype]

theorem body_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial current : State α)
    (k : Nat) (a : CIndex p → α) (ctx : RawSource.Context p initial current)
    (acc : RawSource.HasAcc p a current)
    (ha : ∀ i : AIndex p, RawSource.innerMask p k i.2.1 = Bool.true → RawSource.addressA p initial k i<bounds p.A)
    (hb : ∀ i : BIndex p, RawSource.innerMask p k i.1 = Bool.true → RawSource.addressB p initial k i<bounds p.B) :
    Safety.traceSafeList (backend M D) bounds (RawSource.body p st)
      (current.setReg "k" .int [] (fun _ => (k : Int))) := by
  simp only [RawSource.HasAcc] at acc
  cases st <;> simp [RawSource.body,Source.typed,Source.halfSource,Source.byteSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.dimK,ctx.strideAM,ctx.strideAK,ctx.strideBK,ctx.strideBN,ctx.offk,ctx.am,ctx.bn,acc,backend,dtype]
  all_goals constructor
  all_goals intro i j rest active
  case half.left | e4m3.left =>
    have h := ha (i,j,rest) (by simpa [RawSource.innerMask] using active)
    simpa [RawSource.addressA,RawSource.innerOffset,RawSource.innerMask] using h
  case half.right | e4m3.right =>
    have h := hb (i,j,rest) (by simpa [RawSource.innerMask] using active)
    simpa [RawSource.addressB,RawSource.innerOffset,RawSource.innerMask] using h

theorem epilogue_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial current : State α)
    (ctx : RawSource.Context p initial current)
    (acc : RawSource.HasAcc p (RawSource.accumulation M D p st initial) current)
    (hc : ∀ i : CIndex p, RawSource.active p initial i = Bool.true → RawSource.outputOffset p initial i<bounds p.C) :
    Safety.traceSafeList (backend M D) bounds (RawSource.epilogue p st) current := by
  simp only [RawSource.HasAcc] at acc
  cases st <;> simp [RawSource.epilogue,Source.typed,Source.halfSource,Source.byteSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.strideCM,ctx.strideCN,ctx.am,ctx.bn,ctx.mm,ctx.nn,acc,backend,dtype]

  all_goals intro i j hr hn
  all_goals simpa [RawSource.outputOffset] using hc (i,j,PUnit.unit) (by
    simp [RawSource.active,hr,hn])


#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSafety
