import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardTransport
import bench.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource

namespace VeriTile.Bench.Repairs.DiagSsmTriton.ForwardResults
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

def Forward (kernel : ComputeKernel) (p : ForwardLayout) (length : Nat) : Prop :=
  p.x ≠ p.y → ∀ u : BlockState, ∃ v,
    exec kernel.toAlgKernel u = some v ∧
    (∀ (t : Fin length) (j : Fin p.B), RealReference.active u p.batch p.dim p.B j →
      v.readMem p.y (RealReference.timeOffset u p.batch p.dim p.B t.val j) =
        RealReference.diagSsmForwardSpec u p.s p.x p.l p.batch p.dim p.B t.val j) ∧
    (∀ r o, (r ≠ p.y ∨ ∀ (t : Fin length) (j : Fin p.B),
      RealReference.active u p.batch p.dim p.B j →
      o ≠ RealReference.timeOffset u p.batch p.dim p.B t.val j) →
      v.mem r o = u.mem r o)

theorem correct (p : ForwardLayout) (length : Nat) :
    Forward (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType p length := by
  intro sep u
  obtain ⟨v, he, ho, hf⟩ := RealReference.dssm_runR .triv p.s p.x p.l p.y
    length p.batch p.dim p.B sep u
  have he' := (execR_triv _ _).symm.trans he
  obtain ⟨a,b,ha,hb,obs⟩ := ForwardTransport.runs_related p length u
  have bv : b=v := Option.some.inj (hb.symm.trans he')
  subst b
  refine ⟨a, ?_, ?_, ?_⟩
  · simpa only [exec,ForwardTransport.source,Kernels.ReadableSource.forward_preserved,
      Erasure.forward_body] using ha
  · intro t j active
    exact (obs.readMem_eq _ _).trans (ho t j active)
  · intro r o outside
    exact congrFun (congrFun obs.mem r) o |>.trans (hf r o outside)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.ForwardResults
