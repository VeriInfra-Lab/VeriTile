import bench.tritonbench_g.rmsnorm_fused_llama.PairIO
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original := PairIO.original
abbrev indexCommuted := PairIO.indexCommuted

/-- Both complete FP32 source grids retain runtime epsilon, both loops,
reduction order and the FP16 output round trip. The existing candidate commutes
only the exact integer output-pointer product. All original inputs, native
reductions, complete safe executions, output cells and frames are retained. -/
specification rmsnorm_kernel_equiv (R : Native.Reductions) (X Y W : RegionName)
    (rows columns : Nat) (epsilon : Native.Word) (positive : 0 < columns)
    (limit : columns ≤ 16384) (xy : X ≠ Y) (wy : W ≠ Y) :
    original R X Y W rows columns epsilon ≡[[]] indexCommuted R X Y W rows columns epsilon :=
  PairEquivalence.equivalent R X Y W rows columns epsilon positive limit xy wy

#print_fp_assumptions rmsnorm_kernel_equiv
#axiomsClean rmsnorm_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.FPEquiv
