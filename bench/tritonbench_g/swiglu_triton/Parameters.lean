import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluTriton
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

end VeriTile.Bench.TritonBenchG.SwigluTriton

namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton
inductive Channel where
  | c | da | db
  deriving DecidableEq, Repr

def outputRegion : Channel → RegionName
  | .c => "C" | .da => "A" | .db => "B"

def enabled (backward : Bool) (ch : Channel) : Prop :=
  if backward then ch ≠ .c else ch = .c

instance (b : Bool) (ch : Channel) : Decidable (enabled b ch) := by
  unfold enabled; infer_instance

end VeriTile.Bench.Repairs.SwigluTriton
