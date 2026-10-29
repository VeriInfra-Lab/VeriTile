import bench.repairs.tritonbench_g.parallel_attention.Proofs.Contract

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Complete source launch and host-reduction memory graph. Use forwardKinds
for O/Z and backwardKinds for the combined DQ/DK/DV wrapper. -/
abbrev kernelRun := @combinedRun

end
end VeriTile.Bench.Repairs.ParallelAttention
