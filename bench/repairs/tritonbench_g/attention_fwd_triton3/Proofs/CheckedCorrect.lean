import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.RealCorrect
import bench.repairs.tritonbench_g.attention_fwd_triton3.Contract
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Checked
open VeriTile Triton Host Ownership Raw NativeSchedule
noncomputable section
theorem checked_real (c : Call) (A : FlatAlloc) (pointers : Pointers A)
    (args : Arguments) (_inputs : metadataChecks c args=Bool.true)
    (setup : setupChecks c A pointers=Bool.true) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (s : BlockState) (hu : s.undef=(fun _ _=>0)) :
    Safety c A pointers ∧ (
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState s) final) ∧
      (∀ idx i,active c i → final.readMem (Flat.address c A idx i).1 (Flat.address c A idx i).2=Whole.result c s idx i) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState s) final) := by
  have safety:=admitted c A pointers setup
  refine ⟨safety,?_⟩
  exact Correct.correct c safety.valid A safety.allocated safety.disjoint hc s hu

#axiomsClean checked_real
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Checked
