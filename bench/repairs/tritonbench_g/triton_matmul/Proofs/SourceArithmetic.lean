import bench.repairs.tritonbench_g.triton_matmul.Proofs.Model
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural FPValue
noncomputable section

theorem mask_iff (limit base lane : Nat) : lane<limit-base ↔ base+lane<limit := by omega

theorem clamp_add (limit base lane : Nat) :
    (if lane<limit-base then base else 0)+(if lane<limit-base then lane else 0)=
      (if base+lane<limit then base+lane else 0) := by
  split_ifs <;> omega

#axiomsClean mask_iff
#axiomsClean clamp_add
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
