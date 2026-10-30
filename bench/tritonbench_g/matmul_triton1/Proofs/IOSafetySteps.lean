import bench.tritonbench_g.matmul_triton1.Proofs.FPComplete
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatmulTriton1.IOSafety
open VeriTile.Triton FP.Structural Revision001.Core
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prelude_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend rest P) bounds (prelude p) s := by
  simp [prelude,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]

theorem body_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (k : Nat)
    (h : Invariant P p initial k current)
    (ha : ∀ i : AIndex p, row p initial i.1<p.rows →
      i.2.1.val< inner p-k*p.blockInner → aOffset p initial k i<bounds p.input)
    (hb : ∀ i : BIndex p, i.1.val< inner p-k*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial k i<bounds p.weight) :
    Safety.traceSafeList (backend rest P) bounds (body p)
      (current.setReg "k" .nat [] (fun _ => k)) := by
  simp [body,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,
    Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    h.memory,h.offm,h.offn,h.offk,h.aptrs,h.bptrs,h.acc,backend,State.setReg]
  constructor
  · intro i j u hr hk
    cases u
    simpa only [hr,hk,if_true] using ha (i,j,()) hr hk
  · intro i j u hk hc
    cases u
    simpa only [hk,hc,if_true] using hb (i,j,()) hk hc

theorem epilogue_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (h : Invariant P p initial (steps p) current)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend rest P) bounds (epilogue p) current := by
  simp [epilogue,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,
    Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    h.memory,h.offm,h.offn,h.acc,backend,State.setReg]
  intro i j hr hc'
  simpa [cOffset,hr,hc'] using hc (i,j,()) ⟨hr,hc'⟩

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.MatmulTriton1.IOSafety
