import bench.tritonbench_g.fused_recurrent_retention.Kernels
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardIO
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFiniteIO
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardSurface
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSurface
import VeriTile.Meta.Specification

/-! Independent complete-source real contracts, successful tagged execution,
standard realization and finite-allocation IO/frame for both source kernels.
These statements use the erased selected fp32 profile; native scalar ABI,
fixed-width address behavior and host wrapper reductions are separate. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Correctness
open VeriTile VeriTile.Triton
noncomputable section

def ForwardImplements (kernel : ComputeKernel) (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) : Prop :=
  (∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
    (storeFinal = Bool.true → r.o ≠ r.final) →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      ForwardReal.Post p r hasSeed storeFinal initial final) ∧
  (∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
    (storeFinal = Bool.true → r.o ≠ r.final) →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (ForwardSurface.writes p (launch initial) r storeFinal) (ForwardSurface.expected p r hasSeed initial)) ∧
  (∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
    ∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
      (storeFinal = Bool.true → r.o ≠ r.final) →
      SafetyAccess.Bounds p (launch initial) r hasSeed storeFinal A.extent →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        ForwardIO.Post A p r hasSeed storeFinal initial final)

def BackwardImplements (kernel : ComputeKernel) (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) : Prop :=
  (∀ initial : BlockState, BackwardReal.Disjoint r →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧ BackwardReal.Post p r hasSeed initial final) ∧
  (∀ initial : BlockState, BackwardReal.Disjoint r →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (BackwardSurface.writes p (launch initial) r) (BackwardSurface.expected p r hasSeed initial)) ∧
  (∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
    ∀ initial : BlockState, BackwardReal.Disjoint r → BackwardSafety.Bounds p (launch initial) r hasSeed A.extent →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        BackwardIO.Post A p r hasSeed initial final)

theorem forward_implements (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    ForwardImplements (ForwardSource.kernel p r hasSeed storeFinal swapped) p r hasSeed storeFinal := by
  refine ⟨?_,?_,?_⟩
  · intro initial hq hk hv hf
    exact ForwardReal.source_correct p r hasSeed storeFinal swapped initial hq hk hv hf
  · intro initial hq hk hv hf
    exact ForwardSurface.source_realizes p r hasSeed storeFinal swapped initial hq hk hv hf
  · intro A hd hc initial hq hk hv hf hb
    exact ForwardIO.source_correct A hd hc p r hasSeed storeFinal swapped initial hq hk hv hf hb

theorem backward_implements (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    BackwardImplements (BackwardSource.kernel p r hasSeed swapped) p r hasSeed := by
  refine ⟨?_,?_,?_⟩
  · intro initial hd
    exact BackwardReal.source_correct p r hasSeed swapped initial hd
  · intro initial hd
    exact BackwardSurface.source_realizes p r hasSeed swapped initial hd
  · intro A hd hc initial hr hb
    exact BackwardIO.source_correct A hd hc p r hasSeed swapped initial hr hb

specification fused_recurrent_retention_forward_original_real (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) :
    Spec.Real (ForwardImplements (ForwardSource.kernel p r hasSeed storeFinal Bool.false)
      p r hasSeed storeFinal) := forward_implements p r hasSeed storeFinal Bool.false

specification fused_recurrent_retention_forward_candidate_real (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) :
    Spec.Real (ForwardImplements (ForwardSource.kernel p r hasSeed storeFinal Bool.true)
      p r hasSeed storeFinal) := forward_implements p r hasSeed storeFinal Bool.true

specification fused_recurrent_retention_backward_original_real (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) :
    Spec.Real (BackwardImplements (BackwardSource.kernel p r hasSeed Bool.false)
      p r hasSeed) := backward_implements p r hasSeed Bool.false

specification fused_recurrent_retention_backward_candidate_real (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) :
    Spec.Real (BackwardImplements (BackwardSource.kernel p r hasSeed Bool.true)
      p r hasSeed) := backward_implements p r hasSeed Bool.true

end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Correctness
