import VeriTile.Triton.DSL
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
def inv_ln2 : ℝ := 1.44269504

end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
