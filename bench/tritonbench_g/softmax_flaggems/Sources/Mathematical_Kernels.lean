import VeriTile.Triton.DSL
/-! Storage metadata only. The obsolete infinite-seed executors and their
version-specific execution proofs are retired. All selected whole-grid Real,
FP, native and safety results use the shared repaired finite-seed source. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource
inductive Storage where
  | fp16 | bf16 | fp32 | fp64
  deriving DecidableEq, Repr

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource
