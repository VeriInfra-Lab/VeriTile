import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Pinned ordinary exp accepts only these raw tensor storage types. -/
inductive MathStorage where
  | fp32 | fp64
  deriving DecidableEq, Repr

def MathStorage.storage : MathStorage → Storage
  | .fp32 => .fp32
  | .fp64 => .fp64

end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
