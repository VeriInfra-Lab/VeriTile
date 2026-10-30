import bench.tritonbench_g.lora_expand_gemv.Sources.Mathematical_FPModel
import bench.tritonbench_g.lora_expand_gemv.Sources.Mathematical_LoraExpandGemv

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat outputType)
noncomputable section
set_option maxRecDepth 28000
set_option maxHeartbeats 12000000

inductive InputProfile where
  | halfHalf | halfBfloat | bfloatHalf | bfloatBfloat | singleHalf | singleBfloat
  deriving DecidableEq, Repr

structure Parameters where
  (input weight output : RegionName)
  indices : Region .int
  (inner outputSize splits inputBatchStride inputInnerStride loraStride weightRowStride
    weightInnerStride outputBatchStride outputStride blockRows blockInner : Nat)
  (evenK addInputs castType : Bool)

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

syntax "loraBranches%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(loraBranches%[$inputTy:tritonDType,$weightTy:tritonDType,$outTy:tritonDType]) =>
    `(fun (p : Parameters) =>
      let input_ptr := p.input
      let lora_ptr := p.weight
      let out_ptr := p.output
      let lora_indices := p.indices
      let EVEN_K := p.evenK
      let ADD_INPUTS := p.addInputs
      let CAST_TYPE := p.castType
      triton {
        tl.region input_ptr = $inputTy, lora_ptr = $weightTy, out_ptr = $outTy
        pid_sn = tl.program_id(axis=0)
        cur_batch = tl.program_id(axis=1)
        lora_index = tl.load($((lora_indices : Region .int)) + cur_batch)
        if lora_index != $((-1 : Int)) {
          offset_k = tl.arange(0, $(p.blockInner))
          offset_n = tl.arange(0, $(p.blockRows))
          if EVEN_K {
            tiled_a = tl.load(input_ptr + cur_batch * $(p.inputBatchStride) + offset_k * $(p.inputInnerStride))
          } else {
            tiled_a = tl.load(input_ptr + cur_batch * $(p.inputBatchStride) + offset_k * $(p.inputInnerStride),
              mask=offset_k < $(p.inner), other=0.0)
          }
          split_n_length = tl.cdiv($(p.outputSize), $(p.splits))
          if CAST_TYPE {
            tiled_a = tiled_a.to($weightTy)
          }
          b_ptr = lora_ptr + $(p.loraStride) * lora_index +
            pid_sn * split_n_length * $(p.weightRowStride)
          c_ptr = out_ptr + cur_batch * $(p.outputBatchStride) + pid_sn * split_n_length
          for n in range($(0), split_n_length, $(p.blockRows)) {
            current_n = n + offset_n
            current_n_c = tl.max_contiguous(current_n, $(p.blockRows))
            b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < $(p.inner))
            c_mask = current_n < split_n_length
            tiled_b = tl.load(b_ptr + current_n_c[:, None] * $(p.weightRowStride) +
              offset_k[None, :] * $(p.weightInnerStride), mask=b_ptr_mask, other=0.0)
            if ADD_INPUTS {
              tiled_out = tl.load(c_ptr + current_n * $(p.outputStride), mask=c_mask)
              accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
            } else {
              accumulator = tl.sum(tiled_a * tiled_b, 1)
            }
            tl.store(c_ptr + current_n * $(p.outputStride), accumulator.to($outTy), mask=c_mask)
          }
        }
      })

def source (profile : InputProfile) (out : OutputFormat) (p : Parameters) : ComputeKernel :=
  match profile, out with
  | .halfHalf, .half => loraBranches%[tl.float16,tl.float16,tl.float16] p
  | .halfHalf, .bfloat => loraBranches%[tl.float16,tl.float16,tl.bfloat16] p
  | .halfHalf, .single => loraBranches%[tl.float16,tl.float16,tl.float32] p
  | .halfBfloat, .half => loraBranches%[tl.float16,tl.bfloat16,tl.float16] p
  | .halfBfloat, .bfloat => loraBranches%[tl.float16,tl.bfloat16,tl.bfloat16] p
  | .halfBfloat, .single => loraBranches%[tl.float16,tl.bfloat16,tl.float32] p
  | .bfloatHalf, .half => loraBranches%[tl.bfloat16,tl.float16,tl.float16] p
  | .bfloatHalf, .bfloat => loraBranches%[tl.bfloat16,tl.float16,tl.bfloat16] p
  | .bfloatHalf, .single => loraBranches%[tl.bfloat16,tl.float16,tl.float32] p
  | .bfloatBfloat, .half => loraBranches%[tl.bfloat16,tl.bfloat16,tl.float16] p
  | .bfloatBfloat, .bfloat => loraBranches%[tl.bfloat16,tl.bfloat16,tl.bfloat16] p
  | .bfloatBfloat, .single => loraBranches%[tl.bfloat16,tl.bfloat16,tl.float32] p
  | .singleHalf, .half => loraBranches%[tl.float32,tl.float16,tl.float16] p
  | .singleHalf, .bfloat => loraBranches%[tl.float32,tl.float16,tl.bfloat16] p
  | .singleHalf, .single => loraBranches%[tl.float32,tl.float16,tl.float32] p
  | .singleBfloat, .half => loraBranches%[tl.float32,tl.bfloat16,tl.float16] p
  | .singleBfloat, .bfloat => loraBranches%[tl.float32,tl.bfloat16,tl.bfloat16] p
  | .singleBfloat, .single => loraBranches%[tl.float32,tl.bfloat16,tl.float32] p

end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
