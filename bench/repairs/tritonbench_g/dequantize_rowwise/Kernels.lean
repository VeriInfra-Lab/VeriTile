import bench.tritonbench_g.dequantize_rowwise.Kernels

/-! Selected production route: INT8 weights, FP32 scales and two FP32
multiplications, followed by FP16 output conversion. All four original Python
benchmark cases explicitly select INT8 weights. The canonical readable typed
DSL is reused without a second AST. Signed-64 native indexing is checked in
Proofs/NativeIndex; it has the same natural-index projection. -/
namespace VeriTile.Bench.Repairs.DequantizeRowwise
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DequantizeRowwise

def source (x_ptr state_x output_ptr : RegionName) (inv_127 : ℝ)
    (n_elements BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  triton {
  tl.region x_ptr = tl.int8, state_x = tl.float32, output_ptr = tl.float16
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $((inv_127 : ℝ))
  tl.store(output_ptr + offsets, (output).to(tl.float16), mask=row_mask)
}

def indexCommutedSource (x_ptr state_x output_ptr : RegionName) (inv_127 : ℝ)
    (n_elements BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  triton {
  tl.region x_ptr = tl.int8, state_x = tl.float32, output_ptr = tl.float16
  pid = tl.program_id(axis=0)
  block_start = $(BLOCK_SIZE) * pid
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $((inv_127 : ℝ))
  tl.store(output_ptr + offsets, (output).to(tl.float16), mask=row_mask)
}
def xRegion : RegionName := "x"
def scaleRegion : RegionName := "scale"
def outRegion : RegionName := "out"

structure Parameters where
  rows : Nat
  cols : Nat
  width : Nat
  covers : cols ≤ width

noncomputable def kernel (p : Parameters) : ComputeKernel :=
  source xRegion scaleRegion outRegion (1 / 127) (p.rows * p.cols) p.cols p.width

noncomputable def indexCommutedKernel (p : Parameters) : ComputeKernel :=
  indexCommutedSource xRegion scaleRegion outRegion (1 / 127) (p.rows * p.cols) p.cols p.width

end VeriTile.Bench.Repairs.DequantizeRowwise
