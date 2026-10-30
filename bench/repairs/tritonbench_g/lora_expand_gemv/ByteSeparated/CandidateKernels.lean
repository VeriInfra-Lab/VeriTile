import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural
noncomputable section

/-- Actual source specialization used by the selected wrapper: SPLIT_N=N and
all invoked programs have N>0, hence the split length is one. -/
def unitSplitSource (p : Branches.Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let lora_indices := p.indices
  let EVEN_K := p.evenK
  let ADD_INPUTS := p.addInputs
  let CAST_TYPE := p.castType
  triton {
    tl.region input_ptr = tl.float32, lora_ptr = tl.float16, out_ptr = tl.float32
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
      split_n_length = $(1)
      if CAST_TYPE {
        tiled_a = tiled_a.to(tl.float16)
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
        tl.store(c_ptr + current_n * $(p.outputStride), accumulator.to(tl.float32), mask=c_mask)
      }
    }
  }

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
