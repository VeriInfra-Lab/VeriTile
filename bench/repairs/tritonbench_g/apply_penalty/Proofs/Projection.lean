import bench.repairs.tritonbench_g.apply_penalty.Kernels
import bench.tritonbench_g.apply_penalty.Kernels

namespace VeriTile.Bench.Repairs.ApplyPenalty.Projection
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ApplyPenalty

theorem algorithm (L P F R : RegionName) (I C S : Region .nat) (stride skip B : Nat) :
    (kernel L P F R I C S stride skip B).toAlgKernel =
    (apply_penalty L P F R I C S stride skip B).toAlgKernel := by
  rfl

end VeriTile.Bench.Repairs.ApplyPenalty.Projection
