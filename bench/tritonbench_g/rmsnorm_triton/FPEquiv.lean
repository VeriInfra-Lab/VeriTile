import bench.tritonbench_g.rmsnorm_triton.IO
import bench.tritonbench_g.rmsnorm_triton.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.RmsnormTriton.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []

/-- Complete FP32 source equivalence: the candidate commutes only a natural
row-address product. Native integer power, casts, reduction, reciprocal square
root and every load/store remain identical, with no numerical assumption.
The full physical-grid native observation remains in KernelNativeResults. -/
specification source_candidate_equivalence
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    originalProgram X W O sxb sxm sxk sw sob som sok N B eps ≡[assumptions]
      candidateProgram X W O sxb sxm sxk sw sob som sok N B eps :=
  KernelRewrite.equivalent X W O sxb sxm sxk sw sob som sok N B eps

#print_fp_assumptions source_candidate_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormTriton.FPEquiv
