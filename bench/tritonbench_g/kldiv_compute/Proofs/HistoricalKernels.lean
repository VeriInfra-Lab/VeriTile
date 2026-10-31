import bench.tritonbench_g.kldiv_compute.Sources.Mathematical_Kernels
import bench.tritonbench_g.vector_addition.Sources.Mathematical_Value

/-! The repaired JIT's i64 index casts refine these exact natural indices.
The typed floating suffix is the unchanged original source. Host packing and
copyback are explicit in Proofs.Host. Installed grid-product compatibility is
outside the ordinary launch contract; no segmented-launch change is made. -/
namespace VeriTile.Bench.Repairs.KldivCompute
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters
abbrev source (p : Parameters) :=
  VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel
    .fp32 .fp32 .fp32 Bool.false p.x p.y p.output p.n p.block
end VeriTile.Bench.Repairs.KldivCompute

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n+1023)/1024
noncomputable def mathematical (x y : ℝ) : ℝ := x * Real.log (x/y)
end VeriTile.Bench.Repairs.KldivCompute.Contiguous
