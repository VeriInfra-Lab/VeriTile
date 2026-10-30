import bench.tritonbench_g.reversed_cumsum.Proofs.Source
import VeriTile.Meta.StatementAudit

/-! Local lowering gates. These do not replace the selected-source Python AST
audit, full correctness or native dtype/ABI validation. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.SourceAudit
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

theorem projects (swapped : Bool) (dtype out_dtype : FloatDType) (s z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    (TypedSource.source swapped dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).toAlgorithm? =
      .ok (TypedSource.source swapped dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).toAlgKernel := by
  cases swapped <;> rfl

theorem common_suffix (dtype out_dtype : FloatDType) (s z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    (TypedSource.original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody.drop 4 =
      (TypedSource.compareRight dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody.drop 4 := rfl

#axiomsClean projects
#axiomsClean common_suffix
end VeriTile.Bench.TritonBenchG.ReversedCumsum.SourceAudit
