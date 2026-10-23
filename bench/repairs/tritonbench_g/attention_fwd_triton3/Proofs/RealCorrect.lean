import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Erasure
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Return
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Machine
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Defect

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Correct
open VeriTile.Triton Host Ownership
noncomputable section

/-- Complete FP32 real projection, six actual initial arrays, successful
flat-memory execution of the complete grid, all active O/M/L and exact frame.
Native local numerical obligations are stated separately, not assumed here. -/
theorem correct (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (s : BlockState) (hu : s.undef=(fun _ _=>0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState s) final) ∧
      (∀ idx i,active c i → final.readMem (Flat.address c A idx i).1 (Flat.address c A idx i).2=Whole.result c s idx i) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState s) final := by
  rw [Kernels.real_erasure]
  exact Flat.complete c h A a hd hc s hu

#print axioms correct
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Correct
