import bench.tritonbench_g.matmul_triton1.Proofs.IOSafetySteps

namespace VeriTile.Bench.Repairs.MatmulTriton1.IOSafety
open VeriTile.Triton FP.Structural Revision001.Core
noncomputable section
set_option maxHeartbeats 100000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
attribute [local irreducible] preludeExecution bodyExecution

theorem range_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (k : Nat)
    (h : Invariant P p initial k current)
    (ha : ∀ q (i : AIndex p), row p initial i.1<p.rows →
      i.2.1.val< inner p-q*p.blockInner → aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : BIndex p), i.1.val< inner p-q*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial q i<bounds p.weight) :
    Safety.rangeSafe (backend rest P) bounds "k" k (steps p) 1 (body p) current := by
  have aux : ∀ fuel k (current : State α), steps p-k=fuel →
      Invariant P p initial k current →
      Safety.rangeSafe (backend rest P) bounds "k" k (steps p) 1 (body p) current := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro k current count inv
      by_cases inside : k<steps p
      · rw [Safety.rangeSafe_unfold]
        simp only [show ¬(1 : Nat)=0 by decide,if_false,inside,if_true]
        refine ⟨body_safe rest P p bounds initial current k inv (ha k) (hb k),?_⟩
        rw [(bodyExecution rest P p initial current k inv).property]
        exact ih (steps p-(k+1)) (by omega) (k+1) _ rfl
          (bodyInvariant rest P p initial current k inv)
      · rw [Safety.rangeSafe_unfold]
        simp only [show ¬(1 : Nat)=0 by decide,if_false,inside]
  exact aux _ k current rfl h

#axiomsClean range_safe

theorem loop_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (h : Invariant P p initial 0 current)
    (ha : ∀ q (i : AIndex p), row p initial i.1<p.rows →
      i.2.1.val< inner p-q*p.blockInner → aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : BIndex p), i.1.val< inner p-q*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial q i<bounds p.weight) :
    Safety.stepSafe (backend rest P) bounds (loopStmt p) current := by
  simp [loopStmt,source,typed,ComputeKernel.surfaceBody,Safety.stepSafe_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeExpr_unfold,
    evalOp_unfold,evalExpr,evalComputeOp,numeric,bop,integralFloorDiv,
    Option.bind,ComputeDType.eraseDType]
  exact range_safe rest P p bounds initial current 0 h ha hb

#axiomsClean loop_safe

theorem source_safe (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State α) (hBK : 0<p.blockInner)
    (ha : ∀ q (i : AIndex p), row p initial i.1<p.rows →
      i.2.1.val< inner p-q*p.blockInner → aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : BIndex p), i.1.val< inner p-q*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial q i<bounds p.weight)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.output) :
    TraceSafe (backend rest P) bounds (source p) initial := by
  unfold TraceSafe
  rw [source_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe rest P p bounds initial,?_⟩
  intro start ran
  have equal : start=(preludeExecution rest P p initial).val :=
    Option.some.inj (ran.symm.trans (preludeExecution rest P p initial).property)
  subst start
  obtain ⟨middle,loop,inv⟩ := loopExecution rest P p initial _ 0 (by omega)
    (preludeInvariant rest P p initial)
  rw [List.singleton_append,Safety.traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · exact loop_safe rest P p bounds initial _ (preludeInvariant rest P p initial) ha hb
  · rw [dynamicLoop rest P p _ hBK,loop]
    exact epilogue_safe rest P p bounds initial middle inv hc

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatmulTriton1.IOSafety
