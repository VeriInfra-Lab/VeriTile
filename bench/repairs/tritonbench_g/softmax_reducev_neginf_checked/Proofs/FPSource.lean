import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Kernels
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false


def prelude (p : Parameters) := (source p).surfaceBody.take 12
def loopBody (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (source p).surfaceBody
  match code[12]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def suffix (p : Parameters) := (source p).surfaceBody.drop 13

theorem source_split (p : Parameters) :
    (source p).surfaceBody = prelude p ++
      [.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "cur_batch_seq_len")
        (.constNat p.tokens) (loopBody p)] ++ suffix p := by
  simp [prelude,loopBody,suffix,source,typed,ComputeKernel.surfaceBody]
#axiomsClean source_split
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
