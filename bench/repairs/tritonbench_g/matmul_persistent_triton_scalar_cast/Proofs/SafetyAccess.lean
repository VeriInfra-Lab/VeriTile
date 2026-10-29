import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Execution
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SourceSafety
open VeriTile.Triton FP.Structural Core
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton 
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem int_ite_nat (cond : Prop) [Decidable cond] (a b : Nat) :
    (if cond then (a : Int) else (b : Int))=((if cond then a else b : Nat) : Int) := by
  split <;> rfl

private theorem masked_sum_nat (c d : Prop) [Decidable c] [Decidable d] (a b : Nat) :
    Int.toNat ((if c then (a : Int) else 0)+(if d then (b : Int) else 0))=
      (if c then a else 0)+(if d then b else 0) := by
  split_ifs <;> simp

/-- The Core.inner K loop's two actual typed loads use their reached pointer and
mask registers. Arithmetic primitives do not add memory accesses. -/
theorem inner_safe (P : Source.Arithmetic α α) (p : Core.Parameters)
    (initial current : State α) (tile block : Nat) (h : InnerInvariant P p initial tile block current)
    (bounds : RegionName → Nat)
    (ha : ∀ i : AIndex p,p.problem.k block i.2.1.val<p.problem.K → aOffset p tile block i<bounds "A")
    (hb : ∀ i : BIndex p,p.problem.k block i.1.val<p.problem.K → bOffset p tile block i<bounds "B") :
    Safety.traceSafeList (backend P) bounds (Core.inner p) (current.setReg "ki" .int [] (fun _ => (block : Int))) := by
  rcases p with ⟨⟨M,N,K,SMS,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile
  case fp16 =>
    let p : Core.Parameters := ⟨⟨M,N,K,SMS,.fp16⟩,am,ak,bk,bn,cm,cn⟩
    change Safety.traceSafeList (backend P) bounds (Core.inner p)
      (current.setReg "ki" .int [] (fun _ => (block : Int)))
    have h : InnerInvariant P p initial tile block current := h
    have ha : ∀ i : AIndex p,p.problem.k block i.2.1.val<p.problem.K → aOffset p tile block i<bounds "A" := ha
    have hb : ∀ i : BIndex p,p.problem.k block i.1.val<p.problem.K → bOffset p tile block i<bounds "B" := hb
    simp [p,Core.inner,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,integralFloorDiv,integralMod,backend,matrix,h.offAM,h.offBN,h.offK,h.acc,h.dimM,h.dimN,h.dimK,h.strideAM,h.strideAK,h.strideBK,h.strideBN,h.strideCM,h.strideCN,
      h.memory,aValues,bValues,aOffset,bOffset,readCell,floatValues,dtype,
      Problem.k,clamp,Broadcast.leftIndex,Broadcast.rightIndex,RealMath.signed_tail_mask]
    simp only [← Int.natCast_mul,← Int.natCast_add,int_ite_nat,Int.ofNat_lt,Int.toNat_natCast]
    constructor
    · intro r k hk
      simpa [p,aOffset,Problem.k,clamp,ite_mul, -Nat.cast_mul, -Nat.cast_add, -Nat.cast_ite, ← Int.natCast_mul, ← Int.natCast_add,int_ite_nat,masked_sum_nat] using ha (r,k,PUnit.unit) hk
    · intro k c hk
      simpa [p,bOffset,Problem.k,clamp,ite_mul, -Nat.cast_mul, -Nat.cast_add, -Nat.cast_ite, ← Int.natCast_mul, ← Int.natCast_add,int_ite_nat,masked_sum_nat] using hb (k,c,PUnit.unit) hk
  case f8e4 =>
    let p : Core.Parameters := ⟨⟨M,N,K,SMS,.f8e4⟩,am,ak,bk,bn,cm,cn⟩
    change Safety.traceSafeList (backend P) bounds (Core.inner p)
      (current.setReg "ki" .int [] (fun _ => (block : Int)))
    have h : InnerInvariant P p initial tile block current := h
    have ha : ∀ i : AIndex p,p.problem.k block i.2.1.val<p.problem.K → aOffset p tile block i<bounds "A" := ha
    have hb : ∀ i : BIndex p,p.problem.k block i.1.val<p.problem.K → bOffset p tile block i<bounds "B" := hb
    simp [p,Core.inner,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,integralFloorDiv,integralMod,backend,matrix,h.offAM,h.offBN,h.offK,h.acc,h.dimM,h.dimN,h.dimK,h.strideAM,h.strideAK,h.strideBK,h.strideBN,h.strideCM,h.strideCN,
      h.memory,aValues,bValues,aOffset,bOffset,readCell,floatValues,dtype,
      Problem.k,clamp,Broadcast.leftIndex,Broadcast.rightIndex,RealMath.signed_tail_mask]
    simp only [← Int.natCast_mul,← Int.natCast_add,int_ite_nat,Int.ofNat_lt,Int.toNat_natCast]
    constructor
    · intro r k hk
      simpa [p,aOffset,Problem.k,clamp,ite_mul, -Nat.cast_mul, -Nat.cast_add, -Nat.cast_ite, ← Int.natCast_mul, ← Int.natCast_add,int_ite_nat,masked_sum_nat] using ha (r,k,PUnit.unit) hk
    · intro k c hk
      simpa [p,bOffset,Problem.k,clamp,ite_mul, -Nat.cast_mul, -Nat.cast_add, -Nat.cast_ite, ← Int.natCast_mul, ← Int.natCast_add,int_ite_nat,masked_sum_nat] using hb (k,c,PUnit.unit) hk

theorem suffix_safe (P : Source.Arithmetic α α) (p : Core.Parameters)
    (initial current : State α) (tile : Nat)
    (h : InnerInvariant P p initial tile p.problem.kTiles current) (bounds : RegionName → Nat)
    (ho : ∀ i : CIndex,active p tile i → cOffset p tile i<bounds "C") :
    Safety.traceSafeList (backend P) bounds (tileSuffix p) current := by
  rcases p with ⟨⟨M,N,K,SMS,profile⟩,am,ak,bk,bn,cm,cn⟩
  cases profile
  case fp16 =>
    let p : Core.Parameters := ⟨⟨M,N,K,SMS,.fp16⟩,am,ak,bk,bn,cm,cn⟩
    change Safety.traceSafeList (backend P) bounds (tileSuffix p) current
    have h : InnerInvariant P p initial tile p.problem.kTiles current := h
    have ho : ∀ i : CIndex,active p tile i → cOffset p tile i<bounds "C" := ho
    simp [p,tileSuffix,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,backend,h.offAM,h.offBN,h.offCM,h.offCN,h.acc,h.dimM,h.dimN,h.dimK,h.strideAM,h.strideAK,h.strideBK,h.strideBN,h.strideCM,h.strideCN,
      cOffset,active,floatValues,dtype,pack,Broadcast.leftIndex,Broadcast.rightIndex]
    constructor
    · intro r c hr hc
      simpa [p,cOffset, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add] using ho (r,c,PUnit.unit) ⟨hr,hc⟩
    · split <;> trivial
  case f8e4 =>
    let p : Core.Parameters := ⟨⟨M,N,K,SMS,.f8e4⟩,am,ak,bk,bn,cm,cn⟩
    change Safety.traceSafeList (backend P) bounds (tileSuffix p) current
    have h : InnerInvariant P p initial tile p.problem.kTiles current := h
    have ho : ∀ i : CIndex,active p tile i → cOffset p tile i<bounds "C" := ho
    simp [p,tileSuffix,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,backend,h.offAM,h.offBN,h.offCM,h.offCN,h.acc,h.dimM,h.dimN,h.dimK,h.strideAM,h.strideAK,h.strideBK,h.strideBN,h.strideCM,h.strideCN,
      cOffset,active,floatValues,dtype,pack,Broadcast.leftIndex,Broadcast.rightIndex]
    constructor
    · intro r c hr hc
      simpa [p,cOffset, -Nat.cast_mul, -Nat.cast_add, ← Int.natCast_mul, ← Int.natCast_add] using ho (r,c,PUnit.unit) ⟨hr,hc⟩
    · split <;> trivial

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SourceSafety
