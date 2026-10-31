import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_ValueModel

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] FP.add FP.mul treeValue

end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
