import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure

/-! The repaired FP32 Python body and its current integer-index candidate.
The optional arguments expose the source strides and pointers; the host selects
contiguous row strides. Both semantics share these quotations. -/
namespace VeriTile.Bench.Repairs.SquareMatrix
open VeriTile.Triton

def fp32Kernel (cols B : Nat)
    (input_row_stride : Nat := cols) (output_row_stride : Nat := cols)
    (output_ptr : RegionName := "out") (input_ptr : RegionName := "x") : ComputeKernel :=
  triton {
    tl.ports inputs=[input_ptr], outputs=[output_ptr]
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * $(input_row_stride)
    col_offsets = tl.arange(0, $(B))
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < $(cols), other=-float("inf"))
    square_output = row * row
    output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < $(cols))
  }

def indexCommutedKernel (cols B : Nat)
    (input_row_stride : Nat := cols) (output_row_stride : Nat := cols)
    (output_ptr : RegionName := "out") (input_ptr : RegionName := "x") : ComputeKernel :=
  triton {
    tl.ports inputs=[input_ptr], outputs=[output_ptr]
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + $(input_row_stride) * row_idx
    col_offsets = tl.arange(0, $(B))
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < $(cols), other=-float("inf"))
    square_output = row * row
    output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < $(cols))
  }

end VeriTile.Bench.Repairs.SquareMatrix
