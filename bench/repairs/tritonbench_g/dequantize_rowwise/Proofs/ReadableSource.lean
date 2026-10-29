import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.BodyExecution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DequantizeRowwise.ReadableSource
open VeriTile.Triton
noncomputable section

/-- Calls bind values outside the original pasted function body. -/
theorem original_call (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    source x scale out inv n block lanes =
      (body x scale out block lanes).withScalarArguments (arguments inv n) := rfl

theorem candidate_call (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    indexCommutedSource x scale out inv n block lanes =
      (indexCommutedBody x scale out block lanes).withScalarArguments (arguments inv n) := rfl


#axiomsClean original_call
#axiomsClean candidate_call
#axiomsClean BodyExecution.source_memory
end
end VeriTile.Bench.Repairs.DequantizeRowwise.ReadableSource
