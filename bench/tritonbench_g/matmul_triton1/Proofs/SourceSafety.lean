import bench.tritonbench_g.matmul_triton1.Proofs.SourceSafetySteps
import bench.tritonbench_g.matmul_triton1.Proofs.SourcePublicDomain
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
private theorem body_execution_invariant (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (initial current : State (Word α)) (k : Nat)
    (h : RuntimeInvariant p initial k current) :
    RuntimeInvariant p initial (k+1) (bodyExecution rest P p initial current k h).val :=
  bodyInvariant p initial current k h

attribute [local irreducible] preludeExecution bodyExecution

theorem range_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (bounds : RegionName → Nat) (initial current : State (Word α)) (k : Nat)
    (h : RuntimeInvariant p initial k current)
    (ha : ∀ q (i : Core.AIndex p), Core.row p initial i.1<p.rows →
      i.2.1.val<p.innerSize-q*p.blockInner → Core.aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : Core.BIndex p), i.1.val<p.innerSize-q*p.blockInner →
      Core.column p initial i.2.1<p.columns → Core.bOffset p initial q i<bounds p.weight) :
    Safety.rangeSafe (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds "k" k
      (Core.steps p) 1 (loopBody p) current := by
  have aux : ∀ fuel k (current : State (Word α)), Core.steps p-k=fuel →
      RuntimeInvariant p initial k current →
      Safety.rangeSafe (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds "k" k
        (Core.steps p) 1 (loopBody p) current := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro k current count inv
      by_cases inside : k<Core.steps p
      · rw [Safety.rangeSafe_unfold]
        simp only [show ¬(1 : Nat)=0 by decide,if_false,inside,if_true]
        refine ⟨body_safe rest P p bounds initial current k inv (ha k) (hb k),?_⟩
        rw [(bodyExecution rest P p initial current k inv).property]
        exact ih (Core.steps p-(k+1)) (by omega) (k+1) _ rfl
          (body_execution_invariant rest P p initial current k inv)
      · rw [Safety.rangeSafe_unfold]
        simp only [show ¬(1 : Nat)=0 by decide,if_false,inside]
  exact aux _ k current rfl h

theorem loop_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (bounds : RegionName → Nat) (initial current : State (Word α))
    (h : RuntimeInvariant p initial 0 current)
    (ha : ∀ q (i : Core.AIndex p), Core.row p initial i.1<p.rows →
      i.2.1.val<p.innerSize-q*p.blockInner → Core.aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : Core.BIndex p), i.1.val<p.innerSize-q*p.blockInner →
      Core.column p initial i.2.1<p.columns → Core.bOffset p initial q i<bounds p.weight) :
    Safety.stepSafe (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds (loopStatement p) current := by
  have hk := h.metadata "k_size" p.innerSize (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  rw [loopStatement_eq]
  simp [Safety.stepSafe_unfold,Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeExpr_unfold,
    evalOp_unfold,evalExpr,evalComputeOp,numeric,bop,integralFloorDiv,
    Option.bind,ComputeDType.eraseDType,hk,Core.steps,Core.inner]
  exact range_safe rest P p bounds initial current 0 h ha hb

theorem source_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (bounds : RegionName → Nat) (initial : State (Word α))
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (pid : initial.pids 0 < FiniteIndex.count p)
    (ha : ∀ q (i : Core.AIndex p), Core.row p initial i.1<p.rows →
      i.2.1.val<p.innerSize-q*p.blockInner → Core.aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : Core.BIndex p), i.1.val<p.innerSize-q*p.blockInner →
      Core.column p initial i.2.1<p.columns → Core.bOffset p initial q i<bounds p.weight)
    (hc : ∀ i : Core.CIndex p, Core.active p initial i → Core.cOffset p initial i<bounds p.output) :
    TraceSafe (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) bounds
      (VeriTile.Bench.Repairs.MatmulTriton1.Kernel.source p) initial := by
  have small := scalar_parameters_small p canonical ready
  have pidSmall : initial.pids 0 < 2^64 := by have := ready.launch; omega
  have bmSmall : p.blockRows ≤ 2^64 := by have := ready.blockRows.2; omega
  have bnSmall : p.blockColumns ≤ 2^64 := by have := ready.blockColumns.2; omega
  have bkSmall : p.blockInner ≤ 2^64 := by have := ready.blockInner.2; omega
  have inv := preludeInvariant rest P p initial small pidSmall bmSmall bnSmall bkSmall
  unfold TraceSafe
  rw [source_decomposition,Safety.traceSafeList_append]
  refine ⟨prelude_safe rest P p bounds initial,?_⟩
  intro start ran
  have equal : start=(preludeExecution rest P p initial).val :=
    Option.some.inj (ran.symm.trans (preludeExecution rest P p initial).property)
  subst start
  obtain ⟨middle,loop,finalInv⟩ := loopExecution rest P p initial _ 0 (by omega) inv
  rw [Safety.traceSafeList_unfold]
  refine ⟨loop_safe rest P p bounds initial _ inv ha hb,?_⟩
  rw [dynamicLoop rest P p initial _ 0 inv,loop]
  exact epilogue_safe rest P p bounds initial middle finalInv hc

#axiomsClean range_safe
#axiomsClean loop_safe
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
