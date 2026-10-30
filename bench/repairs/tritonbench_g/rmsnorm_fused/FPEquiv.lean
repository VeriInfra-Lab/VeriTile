import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.RmsnormFused.FPEquiv
open VeriTile Triton FP.ProfiledRewrite SourcePair
open scoped VeriTile.Spec
noncomputable section

/-- Actual repaired runtime-epsilon source versus the existing output-address
candidate. Only row*N becomes N*row in the output pointer. Both passes,
FP32 instructions, int64 PID and epsilon register are preserved, with no
numerical assumption or added domain. Full native grid IO remains auxiliary. -/
specification source_equivalence (X Y W : RegionName) (N B : Nat) :
    original X Y W N B ≡[assumptions] addressCommuted X Y W N B :=
  SourcePair.equivalent X Y W N B

#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.FPEquiv
