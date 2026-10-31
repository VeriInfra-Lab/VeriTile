import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: embedding_triton_kernel.py, embedding_kernel.
Full chunked embedding lookup. Signed int32 token IDs are retained; IDs
outside the local vocabulary produce zero outputs. Data/output precision
matches at fp16, bf16 or fp32. The candidate only commutes the integer product
forming start_n. Loop, load/store masks and explicit padding are unchanged. -/
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.TypedSource
