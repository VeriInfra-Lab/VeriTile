import bench.tritonbench_g.matmul_triton1.Proofs.SourceFullExecution
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem pureList_safe (M : Algebra β) [Inhabited β] (bounds : RegionName → Nat)
    (code : List ComputeStmt) :
    (∀ statement ∈ code, ∀ s, Safety.stepSafe M bounds statement s) →
      ∀ s, Safety.traceSafeList M bounds code s := by
  induction code with
  | nil => intro _ s; simp
  | cons statement rest ih =>
    intro allSafe s
    apply Safety.traceSafeList_cons_intro (allSafe statement (by simp) s)
    intro next _
    exact ih (fun stmt member => allSafe stmt (List.mem_cons.mpr (Or.inr member))) next

theorem prelude_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (bounds : RegionName → Nat) (s : State (Word α)) :
    Safety.traceSafeList (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds (prelude p) s := by
  apply pureList_safe
  simp [prelude,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.source,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.arguments,
    VeriTile.Bench.Repairs.MatmulTriton1.Kernel.body,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,
    ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask]

theorem body_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (bounds : RegionName → Nat) (initial current : State (Word α)) (k : Nat)
    (h : RuntimeInvariant p initial k current)
    (ha : ∀ i : Core.AIndex p, Core.row p initial i.1<p.rows →
      i.2.1.val<p.innerSize-k*p.blockInner → Core.aOffset p initial k i<bounds p.input)
    (hb : ∀ i : Core.BIndex p, i.1.val<p.innerSize-k*p.blockInner →
      Core.column p initial i.2.1<p.columns → Core.bOffset p initial k i<bounds p.weight) :
    Safety.traceSafeList (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds (loopBody p)
      (current.setReg "k" .nat [] (fun _ => k)) := by
  have hm := h.metadata "m_size" p.rows (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  have hn := h.metadata "n_size" p.columns (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  have hk := h.metadata "k_size" p.innerSize (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  have hsx := h.metadata "stride_xk" p.inputInner (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  have hsy := h.metadata "stride_yk" p.weightInner (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  simp [loopBody,loopStatement,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.source,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.arguments,
    VeriTile.Bench.Repairs.MatmulTriton1.Kernel.body,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,
    ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,
    Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    h.memory,h.offm,h.offn,h.offk,h.aptrs,h.bptrs,h.acc,hm,hn,hk,hsx,hsy,
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.add,State.setReg]
  constructor
  · intro i j u hr hk
    cases u
    simpa only [hr,hk,if_true] using ha (i,j,()) hr hk
  · intro i j u hk hc
    cases u
    simpa only [hk,hc,if_true] using hb (i,j,()) hk hc

theorem epilogue_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (bounds : RegionName → Nat) (initial current : State (Word α))
    (h : RuntimeInvariant p initial (Core.steps p) current)
    (hc : ∀ i : Core.CIndex p, Core.active p initial i → Core.cOffset p initial i<bounds p.output) :
    Safety.traceSafeList (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds (epilogue p) current := by
  have hm := h.metadata "m_size" p.rows (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  have hn := h.metadata "n_size" p.columns (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  simp [epilogue,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.source,VeriTile.Bench.Repairs.MatmulTriton1.Kernel.arguments,
    VeriTile.Bench.Repairs.MatmulTriton1.Kernel.body,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,
    ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,
    Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    h.memory,h.offm,h.offn,h.cptrs,h.acc,hm,hn,
    VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra,State.setReg]
  intro i j u hr hc'
  cases u
  simpa [Core.active,Core.cOffset,hr,hc'] using hc (i,j,()) ⟨hr,hc'⟩

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
