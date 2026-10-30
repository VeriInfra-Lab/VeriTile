import bench.repairs.tritonbench_g.parallel_attention.Proofs.FusedGrid

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
open scoped VeriTile.Spec
noncomputable section

/-- The original complete source schedule and its strict assignment-fusion
candidate retain every source stage and its original native compiler algebra. -/
def original (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :=
  ProgramIO.Implementation.original (Fused.io direction d scale) (fun _=>True)

def candidate (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :=
  ProgramIO.Implementation.ofVariant (Fused.variant direction d scale) (fun _=>True)

specification fused_score_equiv (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    original direction d scale ≡[[]] candidate direction d scale := by
  exact ProgramIO.Implementation.of_equivalent (Fused.variant direction d scale)
    (Fused.equivalent direction d scale)

#print_fp_assumptions fused_score_equiv
#axiomsClean fused_score_equiv
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.FPEquiv
