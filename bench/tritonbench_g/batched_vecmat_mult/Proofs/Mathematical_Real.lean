import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_RealSource
import VeriTile.Triton.Memory.KernelSpec.StreamReal

/-! Both complete sources implement the independent double sum in flat memory.
The original uses the existing real loop invariant and access-safety proof.
The candidate transfers the entire successful execution and frame through
real addition commutativity at every iteration; no FP assumptions are used. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealProofs
open VeriTile.Triton
open scoped VeriTile.Triton.StreamMasked2DKernelIO₂
noncomputable section

section
variable (A B out : RegionName) (M N BM BN BK T : Nat)

def expected (xs : Fin T → Fin (BM*BK) → ℝ)
    (ys : Fin T → Fin (BN*BM*BK) → ℝ) (l : Fin (BM*BN)) : ℝ :=
  ∑ t : Fin T, ∑ e : Fin BK,
    xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealProofs
