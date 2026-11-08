import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
open VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.TypedSource
