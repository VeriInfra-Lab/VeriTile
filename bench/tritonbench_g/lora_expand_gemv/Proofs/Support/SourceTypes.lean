import bench.tritonbench_g.lora_expand_gemv.Proofs.Sources.Parameters
import bench.tritonbench_g.lora_expand_gemv.Proofs.Sources.Mathematical_FPTypedSource
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton
open FPValue (OutputFormat)
noncomputable section
inductive InputProfile where
  | halfHalf | halfBfloat | bfloatHalf | bfloatBfloat | singleHalf | singleBfloat
  deriving DecidableEq, Repr

def length (p : Parameters) := (p.outputSize + p.splits - 1) / p.splits

def inputFormat : InputProfile → OutputFormat
  | .halfHalf | .halfBfloat => .half
  | .bfloatHalf | .bfloatBfloat => .bfloat
  | .singleHalf | .singleBfloat => .single

def weightFormat : InputProfile → OutputFormat
  | .halfHalf | .bfloatHalf | .singleHalf => .half
  | .halfBfloat | .bfloatBfloat | .singleBfloat => .bfloat

def effectiveFormat (profile : InputProfile) (p : Parameters) : OutputFormat :=
  if p.castType then weightFormat profile else inputFormat profile

def promote : OutputFormat → OutputFormat → OutputFormat
  | .single, _ | _, .single => .single
  | .half, _ | _, .half => .half
  | .bfloat, .bfloat => .bfloat

def productFormat (profile : InputProfile) (p : Parameters) :=
  promote (effectiveFormat profile p) (weightFormat profile)

def resultFormat (profile : InputProfile) (out : OutputFormat) (p : Parameters) :=
  if p.addInputs then promote (productFormat profile p) out else productFormat profile p

end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
