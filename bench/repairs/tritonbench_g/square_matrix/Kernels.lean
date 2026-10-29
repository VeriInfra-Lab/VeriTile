import bench.tritonbench_g.square_matrix.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Complete selected FP32 sources. The Python body, logical-width arguments,
and historical untyped source alias are retained. -/
namespace VeriTile.Bench.Repairs.SquareMatrix
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("output_ptr",.fp32),("input_ptr",.fp32)]
  scalars := [("input_row_stride",.i32),("output_row_stride",.i32),("n_cols",.i32)]
  constexpr := ["BLOCK_SIZE"]


abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SquareMatrix.square_kernel

def fp32Kernel (cols B : Nat) : ComputeKernel :=
  let output_ptr : RegionName := "out"
  let input_ptr : RegionName := "x"
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [input_row_stride := cols, output_row_stride := cols, n_cols := cols] in
  triton {
    # Each program squares one matrix row.
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row

    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
  }

def indexCommutedKernel (cols B : Nat) : ComputeKernel :=
  let output_ptr : RegionName := "out"
  let input_ptr : RegionName := "x"
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [input_row_stride := cols, output_row_stride := cols, n_cols := cols] in
  triton {
    # Each program squares one matrix row.
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + input_row_stride * row_idx
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row

    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
  }

end VeriTile.Bench.Repairs.SquareMatrix
