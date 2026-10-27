import VeriTile.Triton.Core.Ast

/-! The historical manual quotation inserted two same-FP32 output casts.
This proof-only view records the exact syntactic difference from the current
raw Python body. It is not a theorem that arbitrary casts are identities;
actual execution/Real/native proofs use the current source and store values. -/
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.LegacyStoreView
open VeriTile.Triton

def stripCast {dtype shape} (op : Op dtype shape) : Op dtype shape :=
  match op with
  | .castFloat .real .real inner => inner
  | other => other

def value {dtype shape} (expr : ComputeExpr dtype shape) : ComputeExpr dtype shape :=
  match expr with
  | .compute (.alg .fp32 inner) => .compute (.alg .fp32 (stripCast inner))
  | other => other

def store (stmt : ComputeStmt) : ComputeStmt :=
  match stmt with
  | .store dtype sh memory expr mask =>
      .store dtype sh memory (value expr) mask
  | other => other

def loop (stmt : ComputeStmt) : ComputeStmt :=
  match stmt with
  | .forRangeDyn index start stop stride body =>
      .forRangeDyn index start stop stride (body.map store)
  | other => other

def current : ComputeKernel → ComputeKernel
  | .mk inputs outputs body => .mk inputs outputs (body.map loop)

end VeriTile.Bench.Repairs.FastRopeEmbedding.LegacyStoreView
