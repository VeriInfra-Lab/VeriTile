import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardFinal
import bench.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource

namespace VeriTile.Bench.Repairs.DiagSsmTriton.BackwardResults
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

/-- No pair of read-only inputs is required to be disjoint. -/
structure BackwardSeparation (p : BackwardLayout) : Prop where
  initial : p.gx ≠ p.s
  output : p.gx ≠ p.y
  gradient : p.gx ≠ p.gy
  transition : p.gx ≠ p.l
  initialGradient : p.gs ≠ p.gx
  transitionGradient : p.gl ≠ p.gx
  terminal : p.gl ≠ p.gs

def Backward (kernel : ComputeKernel) (p : BackwardLayout) (length : Nat) : Prop :=
  BackwardSeparation p → ∀ u : BlockState, ∃ v,
    exec kernel.toAlgKernel u = some v ∧
    (∀ (t : Fin length) (j : Fin p.B), RealReference.active u p.batch p.dim p.B j →
      v.readMem p.gx (RealReference.timeOffset u p.batch p.dim p.B t.val j) =
        RealReference.diagSsmBackwardGradXSpec u p.l p.gy p.batch p.dim p.B length t.val j) ∧
    (∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
      v.readMem p.gs (RealReference.colOffset u p.B j) =
        RealReference.diagSsmBackwardGradSAfter u p.l p.gy p.batch p.dim p.B length j length) ∧
    (∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
      v.readMem p.gl (RealReference.colOffset u p.B j) =
        RealReference.diagSsmBackwardGradLambdaAfter u p.s p.l p.y p.gy
          p.batch p.dim p.B length j length) ∧
    (∀ r o,
      (r ≠ p.gx ∨ ∀ (t : Fin length) (j : Fin p.B),
        RealReference.active u p.batch p.dim p.B j →
        o ≠ RealReference.timeOffset u p.batch p.dim p.B t.val j) →
      (r ≠ p.gs ∨ ∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
        o ≠ RealReference.colOffset u p.B j) →
      (r ≠ p.gl ∨ ∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
        o ≠ RealReference.colOffset u p.B j) → v.mem r o = u.mem r o)

theorem correct (p : BackwardLayout) (length : Nat) :
    Backward (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length := by
  intro sep u
  obtain ⟨v,he,hx,hs,hl,hf⟩ := RealReference.dssmb_runR .triv p.s p.l p.y p.gs p.gx p.gl p.gy
    length p.batch p.dim p.B sep.initial sep.output sep.gradient sep.transition
    sep.initialGradient sep.transitionGradient sep.terminal u
  have he' := (execR_triv _ _).symm.trans he
  obtain ⟨a,b,ha,hb,obs⟩ := BackwardFinal.runs_related p length u
  have bv : b=v := Option.some.inj (hb.symm.trans he')
  subst b
  refine ⟨a,?_,?_,?_,?_,?_⟩
  · simpa only [exec,BackwardTransport.source,Kernels.ReadableSource.backward_preserved,
      Erasure.backward_body] using ha
  · intro t j active
    exact (obs.readMem_eq _ _).trans (hx t j active)
  · intro j active
    exact (obs.readMem_eq _ _).trans (hs j active)
  · intro j active
    exact (obs.readMem_eq _ _).trans (hl j active)
  · intro r o ox os ol
    exact (congrFun (congrFun obs.mem r) o).trans (hf r o ox os ol)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.BackwardResults
