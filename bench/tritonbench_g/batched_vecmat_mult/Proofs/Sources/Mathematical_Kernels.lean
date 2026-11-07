import bench.tritonbench_g.batched_vecmat_mult.Kernels

/-! Compatibility names for the canonical repaired source. No second kernel
implementation is retained; execution proofs consume the pasted Python body. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.TypedSource
abbrev original := VeriTile.Bench.Repairs.BatchedVecmatMult.sourceKernel
abbrev blockFirst := VeriTile.Bench.Repairs.BatchedVecmatMult.blockFirstKernel
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.TypedSource
