import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_ValueComplete

/-! Complete original fp32 software-value contract. The original source
executes for all natural model parameters and every initial total typed memory.
No successful-execution premise, output-equality premise, real-rounding law or
scalar algebraic identity is required. Native block/address validity, allocation,
hardware numerical binding and device scheduling are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
