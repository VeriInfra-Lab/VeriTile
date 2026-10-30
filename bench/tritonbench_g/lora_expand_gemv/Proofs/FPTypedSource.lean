import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.TermModel
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

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

def source_halfHalf (out_dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let lora_indices := p.indices
  triton {
  tl.region input_ptr = tl.float16, lora_ptr = tl.float16, out_ptr = OUT_DTYPE
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(p.blockInner))
  offset_n = tl.arange(0, $(p.blockRows))
  tiled_a = tl.load(input_ptr + cur_batch * $(p.inputBatchStride) + offset_k * $(p.inputInnerStride),
    mask=offset_k < $(p.inner), other=0.0)
  b_ptr = lora_ptr + $(p.loraStride) * lora_index +
    pid_sn * $(p.length) * $(p.weightRowStride)
  c_ptr = out_ptr + cur_batch * $(p.outputBatchStride) + pid_sn * $(p.length)
  for n in range($(0), $(p.length), $(p.blockRows)) {
    current_n = n + offset_n
    tiled_b = tl.load(b_ptr + current_n[:, None] * $(p.weightRowStride) +
      offset_k[None, :] * $(p.weightInnerStride),
      mask=(current_n[:, None] < $(p.length)) and (offset_k[None, :] < $(p.inner)), other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(p.outputStride), accumulator.to(OUT_DTYPE),
      mask=current_n < $(p.length))
  }
}

def source_halfBfloat (out_dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let lora_indices := p.indices
  triton {
  tl.region input_ptr = tl.float16, lora_ptr = tl.bfloat16, out_ptr = OUT_DTYPE
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(p.blockInner))
  offset_n = tl.arange(0, $(p.blockRows))
  tiled_a = tl.load(input_ptr + cur_batch * $(p.inputBatchStride) + offset_k * $(p.inputInnerStride),
    mask=offset_k < $(p.inner), other=0.0)
  b_ptr = lora_ptr + $(p.loraStride) * lora_index +
    pid_sn * $(p.length) * $(p.weightRowStride)
  c_ptr = out_ptr + cur_batch * $(p.outputBatchStride) + pid_sn * $(p.length)
  for n in range($(0), $(p.length), $(p.blockRows)) {
    current_n = n + offset_n
    tiled_b = tl.load(b_ptr + current_n[:, None] * $(p.weightRowStride) +
      offset_k[None, :] * $(p.weightInnerStride),
      mask=(current_n[:, None] < $(p.length)) and (offset_k[None, :] < $(p.inner)), other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(p.outputStride), accumulator.to(OUT_DTYPE),
      mask=current_n < $(p.length))
  }
}

def source_bfloatHalf (out_dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let lora_indices := p.indices
  triton {
  tl.region input_ptr = tl.bfloat16, lora_ptr = tl.float16, out_ptr = OUT_DTYPE
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(p.blockInner))
  offset_n = tl.arange(0, $(p.blockRows))
  tiled_a = tl.load(input_ptr + cur_batch * $(p.inputBatchStride) + offset_k * $(p.inputInnerStride),
    mask=offset_k < $(p.inner), other=0.0)
  b_ptr = lora_ptr + $(p.loraStride) * lora_index +
    pid_sn * $(p.length) * $(p.weightRowStride)
  c_ptr = out_ptr + cur_batch * $(p.outputBatchStride) + pid_sn * $(p.length)
  for n in range($(0), $(p.length), $(p.blockRows)) {
    current_n = n + offset_n
    tiled_b = tl.load(b_ptr + current_n[:, None] * $(p.weightRowStride) +
      offset_k[None, :] * $(p.weightInnerStride),
      mask=(current_n[:, None] < $(p.length)) and (offset_k[None, :] < $(p.inner)), other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(p.outputStride), accumulator.to(OUT_DTYPE),
      mask=current_n < $(p.length))
  }
}

def source_bfloatBfloat (out_dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let lora_indices := p.indices
  triton {
  tl.region input_ptr = tl.bfloat16, lora_ptr = tl.bfloat16, out_ptr = OUT_DTYPE
  pid_sn = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_k = tl.arange(0, $(p.blockInner))
  offset_n = tl.arange(0, $(p.blockRows))
  tiled_a = tl.load(input_ptr + cur_batch * $(p.inputBatchStride) + offset_k * $(p.inputInnerStride),
    mask=offset_k < $(p.inner), other=0.0)
  b_ptr = lora_ptr + $(p.loraStride) * lora_index +
    pid_sn * $(p.length) * $(p.weightRowStride)
  c_ptr = out_ptr + cur_batch * $(p.outputBatchStride) + pid_sn * $(p.length)
  for n in range($(0), $(p.length), $(p.blockRows)) {
    current_n = n + offset_n
    tiled_b = tl.load(b_ptr + current_n[:, None] * $(p.weightRowStride) +
      offset_k[None, :] * $(p.weightInnerStride),
      mask=(current_n[:, None] < $(p.length)) and (offset_k[None, :] < $(p.inner)), other=0.0)
    accumulator = tl.sum(tiled_a * tiled_b, 1)
    tl.store(c_ptr + current_n * $(p.outputStride), accumulator.to(OUT_DTYPE),
      mask=current_n < $(p.length))
  }
}

def source (profile : InputProfile) (out : OutputFormat) (p : Parameters) : ComputeKernel :=
  match profile with
  | .halfHalf => source_halfHalf (outputType out) p
  | .halfBfloat => source_halfBfloat (outputType out) p
  | .bfloatHalf => source_bfloatHalf (outputType out) p
  | .bfloatBfloat => source_bfloatBfloat (outputType out) p

def prelude (profile : InputProfile) (out : OutputFormat) (p : Parameters) :=
  (source profile out p).surfaceBody.take 8

def body (profile : InputProfile) (out : OutputFormat) (p : Parameters) : List ComputeStmt :=
  match (source profile out p).surfaceBody[8]? with
  | some (ComputeStmt.forRange _ _ _ _ b) => b
  | _ => []

theorem source_decomposition (profile : InputProfile) (out : OutputFormat) (p : Parameters) :
    (source profile out p).surfaceBody = prelude profile out p ++
      [.forRange "n" 0 p.length p.blockRows (body profile out p)] := by
  cases profile <;> rfl

#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
