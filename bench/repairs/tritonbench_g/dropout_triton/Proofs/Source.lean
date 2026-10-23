import bench.repairs.tritonbench_g.dropout_triton.Kernels
import bench.tritonbench_g.dropout_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DropoutTriton.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DropoutTriton

/-- The selected readable source is exactly the source used by the existing
per-program proofs, before any mathematical precision erasure. -/
theorem original (X K O : RegionName) (n B : Nat) (p : ℝ) :
    sourceKernel X K O n B p = TypedSource.source .fp32 .int32 Bool.false X K O n B p := rfl

theorem commuted (X K O : RegionName) (n B : Nat) (p : ℝ) :
    commutedSourceKernel X K O n B p = TypedSource.source .fp32 .int32 Bool.true X K O n B p := rfl

#axiomsClean original
#axiomsClean commuted
end VeriTile.Bench.Repairs.DropoutTriton.Source
