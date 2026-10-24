import bench.repairs.tritonbench_g.square_matrix.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only preservation witnesses. -/
namespace VeriTile.Bench.Repairs.SquareMatrix.SourcePreservation
open VeriTile.Triton

def previous_fp32Kernel (cols B : Nat) : ComputeKernel :=
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

def previous_indexCommutedKernel (cols B : Nat) : ComputeKernel :=
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


theorem fp32Kernel_preserved (cols B : Nat) : fp32Kernel cols B = previous_fp32Kernel cols B := rfl
theorem indexCommutedKernel_preserved (cols B : Nat) : indexCommutedKernel cols B = previous_indexCommutedKernel cols B := rfl

#axiomsClean fp32Kernel_preserved
#axiomsClean indexCommutedKernel_preserved
end VeriTile.Bench.Repairs.SquareMatrix.SourcePreservation
