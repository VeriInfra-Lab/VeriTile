import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton
/-- Tested Python integer mask specializations. Both use signed integer memory. -/
inductive MaskStorage where
  | int32 | int64
  deriving DecidableEq, Repr

/-- Floating storage profiles whose runtime fp32 alpha promotes arithmetic to fp32. -/
inductive FloatStorage where
  | fp32 | fp16 | bf16
  deriving DecidableEq, Repr

end VeriTile.Bench.TritonBenchG.MaskedAddCuda
