import bench.tritonbench_g.square_matrix.Sources.Mathematical_Kernels

/-! Complete selected FP32 sources. The Python body, logical-width arguments,
and historical untyped source alias are retained. -/
namespace VeriTile.Bench.Repairs.SquareMatrix
open VeriTile.Triton

abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SquareMatrix.square_kernel

def fp32Kernel (cols B : Nat) : ComputeKernel :=
  let output_ptr : RegionName := "out"
  let input_ptr : RegionName := "x"
  triton {
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * $(cols)
    col_offsets = tl.arange(0, $(B))
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < $(cols), other=-float("inf"))
    square_output = row * row
    output_row_start_ptr = output_ptr + row_idx * $(cols)
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < $(cols))
  }

def indexCommutedKernel (cols B : Nat) : ComputeKernel :=
  let output_ptr : RegionName := "out"
  let input_ptr : RegionName := "x"
  triton {
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + $(cols) * row_idx
    col_offsets = tl.arange(0, $(B))
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < $(cols), other=-float("inf"))
    square_output = row * row
    output_row_start_ptr = output_ptr + row_idx * $(cols)
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < $(cols))
  }

end VeriTile.Bench.Repairs.SquareMatrix
