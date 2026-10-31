import VeriTile.Triton.DSL

/-! Pure storage tags used by local numerical-operation lemmas.
The retired complete kernels are gone; actual sources live in Sources/Kernels. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedSource
open VeriTile.Triton

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedSource
