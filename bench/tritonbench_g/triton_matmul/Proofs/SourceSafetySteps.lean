import bench.tritonbench_g.triton_matmul.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prelude_safe (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (bounds : RegionName→Nat) (s : State α) :
    Safety.traceSafeList (backend M D) bounds (VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.prelude p st) s := by
  cases st <;> simp [VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.prelude,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,
    ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericLt,numericLe,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,dtype]

theorem body_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName→Nat) (initial current : State α) (k : Nat) (a : CIndex p→α)
    (ctx : Context p initial current) (acc : VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.HasAcc p a current)
    (ha : ∀ i : AIndex p,i.2.1.val+k*BK p<p.K→aOffset p initial k i<bounds p.A)
    (hb : ∀ i : BIndex p,i.1.val+k*BK p<p.K→bOffset p initial k i<bounds p.B) :
    Safety.traceSafeList (backend M D) bounds (iteration p st)
      (current.setReg "k" .int [] (fun _=>(k : Int))) := by
  simp only [VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.HasAcc] at acc
  cases st <;> simp [iteration,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,
    ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericLt,numericLe,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,dtype,
    ctx.metadata.2.2.1,ctx.metadata.2.2.2.1,ctx.metadata.2.2.2.2.1,
    ctx.metadata.2.2.2.2.2.1,ctx.metadata.2.2.2.2.2.2.1,
    ctx.memory,ctx.offk,ctx.am,ctx.bn,acc,backend]
  all_goals constructor
  all_goals intro i j rest active
  case half.left | e4m3.left =>
    have hcond : j.val+k*BK p<p.K := by
      have ai:=active
      simp only [←Int.natCast_mul] at ai
      change (_ : Int)<(p.K : Int)-(k*BK p : Nat) at ai
      omega
    have h:=ha (i,j,rest) hcond
    simp only [if_pos active]
    simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    convert h using 1 <;> dsimp [aOffset] <;> ring
  case half.right | e4m3.right =>
    have hcond : i.val+k*BK p<p.K := by
      have ai:=active
      simp only [←Int.natCast_mul] at ai
      change (_ : Int)<(p.K : Int)-(k*BK p : Nat) at ai
      omega
    have h:=hb (i,j,rest) hcond
    simp only [if_pos active]
    simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
    convert h using 1 <;> dsimp [bOffset] <;> ring

theorem epilogue_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName→Nat) (initial current : State α)
    (ctx : Context p initial current)
    (acc : VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.HasAcc p (accumulation M D p st initial) current)
    (hc : ∀ i : CIndex p,active p initial i→cOffset p initial i<bounds p.C) :
    Safety.traceSafeList (backend M D) bounds (VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.epilogue p st) current := by
  simp only [VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.HasAcc] at acc
  cases st <;> simp [VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.epilogue,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,
    ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericLe,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,dtype,
    ctx.metadata.2.2.2.2.2.2.2.1,ctx.metadata.2.2.2.2.2.2.2.2,
    ctx.memory,ctx.am,ctx.bn,ctx.mm,ctx.nn,acc,backend]
  all_goals intro i j hr hn
  all_goals simpa only [cOffset,aRow,bCol,hr,hn,↓reduceIte,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast] using hc (i,j,PUnit.unit) ⟨hr,hn⟩

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
