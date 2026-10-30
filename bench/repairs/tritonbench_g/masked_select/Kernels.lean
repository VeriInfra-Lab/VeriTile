import bench.tritonbench_g.masked_select.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Complete unchanged selected Python function body. All invocation metadata
is outside. The existing prefix channel is the natural-valued mathematical
projection of signed-int64 cumsum storage; the existing IO/native refinement
continues to prove nonnegativity and its storage bounds. -/
namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("inp_ptr",.fp32), ("select_mask_ptr",.boolean),
    ("prefix_sum_ptr",.i64), ("out_ptr",.fp32)]
  scalars := [("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE"]
  naturalIntegerPointers := ["prefix_sum_ptr"]
  readPorts := ["inp_ptr", "select_mask_ptr", "prefix_sum_ptr"]
  writePorts := ["out_ptr"]

def kernel (p : VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy.Parameters) : ComputeKernel :=
  let inp_ptr := p.input
  let out_ptr := p.output
  let select_mask_ptr := p.selection
  let prefix_sum_ptr := p.prefixSum
  let BLOCK_SIZE := p.width
  with_triton_call callSignature call_args [n_elements := p.size] in
  triton {
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0) - 1

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))
  }

abbrev configs := VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.configs
end VeriTile.Bench.Repairs.MaskedSelect
