import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairNumeric
open VeriTile.Triton

/-- Compatibility name for the shared all-word concrete FMA certificate. -/
theorem fma_mul_comm {f : FP.Format} (cfg : FP.Config) (a b z : FP.Value f) :
    FP.fma cfg a b z = FP.fma cfg b a z :=
  FP.ScalarOpsLaws.fma_mul_comm cfg a b z

end VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairNumeric
