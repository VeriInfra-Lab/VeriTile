import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1
open VeriTile.Triton

/-! Storage labels used by the independent mathematical model. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1
