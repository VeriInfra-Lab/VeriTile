import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Kernels
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.RealCorrect

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RawKernel
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
noncomputable section

/-- Raw shared-source grid execution, all six-input mathematical outputs,
padded O/M/L writes and the complete exterior flat-memory frame. -/
theorem whole_real (c : Call) (valid : Valid c) (A : FlatAlloc)
    (allocated : Allocated c A.extent) (disjoint : A.Disjoint)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (input : BlockState) (defined : input.undef=(fun _ _=>0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState input) final) ∧
      (∀ idx i,active c i → final.readMem (Flat.address c A idx i).1 (Flat.address c A idx i).2=
        Whole.result c input idx i) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState input) final := by
  rw [Kernels.source_eq_inherited]
  exact Correct.correct c valid A allocated disjoint closed input defined

theorem correct (c : Call) (valid : Valid c) (A : FlatAlloc)
    (allocated : Allocated c A.extent) (disjoint : A.Disjoint)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (input : BlockState) (defined : input.undef=(fun _ _=>0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState input) final) ∧
      (∀ idx i,active c i → final.readMem (Flat.address c A idx i).1 (Flat.address c A idx i).2=
        Whole.result c input idx i) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState input) final :=
  whole_real c valid A allocated disjoint closed input defined

#axiomsClean whole_real
#axiomsClean correct
#specNonCircular Whole.result avoiding [whole_real,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RawKernel
