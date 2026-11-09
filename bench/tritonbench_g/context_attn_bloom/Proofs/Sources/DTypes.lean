import VeriTile.Triton.DSL.CallSignature
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
end VeriTile.Bench.TritonBenchG.ContextAttnBloom
