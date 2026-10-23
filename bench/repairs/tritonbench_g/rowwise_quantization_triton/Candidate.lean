import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Candidate
open VeriTile.Triton

/-- Store the completed FP16 maximum before quantization, shortening its live
range across the divide, scale, llrint and INT8 store. Numerical operations,
source conversions and both output channels are otherwise unchanged. -/
def source (x_ptr output_ptr output_maxs : RegionName)
    (columns width : Nat) : ComputeKernel := triton {
  tl.region x_ptr = tl.float32, output_ptr = tl.int8, output_maxs = tl.float16
  pid = tl.program_id(axis=0)
  block_start = pid * $(columns)
  arange = tl.arange(0, $(width))
  offsets = block_start + arange
  row_mask = arange < $(columns)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  tl.store(output_maxs + pid, max_val.to(tl.float16))
  safe_max = tl.where(max_val == 0.0, 1.0, max_val)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x / safe_max))
  tl.store(output_ptr + offsets, output.to(tl.int8), mask=row_mask)
}
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Candidate
