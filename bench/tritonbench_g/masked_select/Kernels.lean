import bench.tritonbench_g.masked_select.Sources.Mathematical_Kernels

/-! The checked wrapper preserves the complete original kernel and all twenty
source autotune configurations. Payload storage is the selected FP32 profile. -/
namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Triton

def kernel (p : VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy.Parameters) : ComputeKernel :=
  let inp_ptr := p.input
  let out_ptr := p.output
  let select_mask_ptr := p.selection
  let prefix_sum_ptr := p.prefixSum
  triton {
    tl.ports inputs=[inp_ptr, select_mask_ptr.cast, prefix_sum_ptr.cast], outputs=[out_ptr]
    tl.region inp_ptr = tl.float32, out_ptr = tl.float32
    pid = tl.program_id(axis=0)
    offsets = pid * $(p.width) + tl.arange(0, $(p.width))
    mask = offsets < $(p.size)
    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets,
      mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets,
      mask=mask, other=0.0) - $(1)
    tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
  }

abbrev configs := VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.configs
end VeriTile.Bench.Repairs.MaskedSelect
