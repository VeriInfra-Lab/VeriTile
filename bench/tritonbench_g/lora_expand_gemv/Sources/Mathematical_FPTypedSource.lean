import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.TermModel
import bench.tritonbench_g.quantize_kv_transform.Sources.Mathematical_MixedBits
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

/-! The original complete accepted active body: ADD_INPUTS=false, CAST_TYPE=false,
masked input load, arbitrary output-loop length. Host sentinel/grid dispatch
remains outside this stage. Storage choices are explicit, including mixed
half/bfloat16 multiplication; no scalar Real evaluator is used. -/
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive InputProfile where
  | halfHalf | halfBfloat | bfloatHalf | bfloatBfloat
  deriving DecidableEq, Repr
inductive OutputFormat where
  | half | bfloat | single
  deriving DecidableEq, Repr

def inputType : InputProfile → FloatDType
  | .halfHalf | .halfBfloat => .fp16
  | .bfloatHalf | .bfloatBfloat => .bf16
def weightType : InputProfile → FloatDType
  | .halfHalf | .bfloatHalf => .fp16
  | .halfBfloat | .bfloatBfloat => .bf16
def productType : InputProfile → FloatDType
  | .bfloatBfloat => .bf16
  | _ => .fp16
def outputType : OutputFormat → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .fp32

structure Parameters where
  (input weight output : RegionName)
  indices : Region .nat
  (inner length inputBatchStride inputInnerStride loraStride weightRowStride
    weightInnerStride outputBatchStride outputStride blockRows blockInner : Nat)

end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
