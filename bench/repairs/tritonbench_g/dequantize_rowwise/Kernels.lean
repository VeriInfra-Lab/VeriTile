import VeriTile.Triton.DSL.CallSignature

/-! The quoted body is copied directly from the selected Python JIT. Pointer
storage and scalar typing are supplied by the call interface, outside the body.
Only BLOCK_SIZE and P2 are constexpr; inv_127 and n_elements are runtime values. -/
namespace VeriTile.Bench.Repairs.DequantizeRowwise
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.i8),("state_x",.fp32),("output_ptr",.fp16)]
  scalars := [("inv_127",.fp32),("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE","P2"]

/-- The unmodified selected Python function body. -/
def body (x_ptr state_x output_ptr : RegionName) (BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
  pid = tl.program_id(axis=0).to(tl.int64)
  block_start = pid * BLOCK_SIZE
  arange = tl.arange(0, P2).to(tl.int64)
  offsets = block_start + arange
  row_mask = arange < BLOCK_SIZE
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * inv_127
  tl.store(output_ptr + offsets, output, mask=row_mask)
}

/-- Candidate: commute only the signed integer row-start multiplication. -/
def indexCommutedBody (x_ptr state_x output_ptr : RegionName) (BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
  pid = tl.program_id(axis=0).to(tl.int64)
  block_start = BLOCK_SIZE * pid
  arange = tl.arange(0, P2).to(tl.int64)
  offsets = block_start + arange
  row_mask = arange < BLOCK_SIZE
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * inv_127
  tl.store(output_ptr + offsets, output, mask=row_mask)
}

/-- Explicit call bindings, separate from the source quotation. The unused
n_elements parameter is retained; scalar FP32 rounding occurs at the call. -/
noncomputable def arguments (inv_127 : ℝ) (n_elements : Nat) : List ScalarArgument :=
  [ScalarArgument.fp32 "inv_127" inv_127, ScalarArgument.integer "n_elements" n_elements]

noncomputable def source (x_ptr state_x output_ptr : RegionName) (inv_127 : ℝ)
    (n_elements BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  (body x_ptr state_x output_ptr BLOCK_SIZE P2).withScalarArguments (arguments inv_127 n_elements)

noncomputable def indexCommutedSource (x_ptr state_x output_ptr : RegionName) (inv_127 : ℝ)
    (n_elements BLOCK_SIZE P2 : Nat) : ComputeKernel :=
  (indexCommutedBody x_ptr state_x output_ptr BLOCK_SIZE P2).withScalarArguments (arguments inv_127 n_elements)

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
