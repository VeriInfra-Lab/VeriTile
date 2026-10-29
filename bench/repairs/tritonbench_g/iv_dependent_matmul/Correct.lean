import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Correct
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
noncomputable section

specification checks_iff (r : Request) (defaultFP32 cuda : Bool) :
    Contract.check r defaultFP32 cuda=.ok () ↔ Contract.Preconditions r defaultFP32 cuda :=
  Contract.check_iff r defaultFP32 cuda

/-- Unrestricted Real inputs are the actual random-allocation snapshots. The
checked call includes all three allocations, empty return or complete finite
source/grid, exact C identity/metadata, every result, and the caller frame. -/
specification matmul_checked_real {Device Error : Type} (api : Wrapper.Calls ℝ Device Error)
    (r : Request) (before : Wrapper.Heap ℝ) (defaultFP32 cuda : Bool)
    (h : Contract.Preconditions r defaultFP32 cuda) (d : Device)
    (p : Wrapper.Prepared ℝ r before) (hp : Wrapper.PreparedBy api r before d p)
    (hl : CheckedHost.LaunchReady api d p) :
    VeriTile.Spec.Real (∃ out,
      CheckedHost.call Execution.ideal .source api r before defaultFP32 cuda=some out ∧
      CheckedHost.Post p out (RealTarget.mathematical r.inner
        (Ideal.inputA p.request p.a.after.state) (Ideal.inputB p.request p.b.after.state))) := by
  exact ⟨_,CheckedHost.call_prepared Execution.ideal .source api r before defaultFP32 cuda h d p hp hl,
    CheckedHost.real_post p h.2.2 .source⟩

#axiomsClean checks_iff
#axiomsClean matmul_checked_real
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Correct
