import bench.repairs.tritonbench_g.kldiv_compute.Kernels
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Program

/-! Revision002 reuses the frozen, checked original typed source and complete
per-program/evolving-grid proof. Only the wrapper changes. Both source kernels
have the same exact floating suffix and widened integer-address pipeline. -/
namespace VeriTile.Bench.Repairs.KldivCompute.Revision002
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.Repairs.KldivCompute.Parameters
abbrev source := VeriTile.Bench.Repairs.KldivCompute.source
end VeriTile.Bench.Repairs.KldivCompute.Revision002
