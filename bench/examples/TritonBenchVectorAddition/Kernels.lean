import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-!
TritonBenchVectorAddition: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]

/-- The fp32 kernel; correctness interprets its mathematical projection. -/
def originalKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let y_ptr : RegionName := "y"
  let output_ptr : RegionName := "output"
  let BLOCK_SIZE : Nat := blockSize
  with_triton_call callSignature call_args [n_elements := nElements] in
  triton {
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- The sole rewrite is output = y + x. Addresses/masks are unchanged. -/
def optimizedKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let y_ptr : RegionName := "y"
  let output_ptr : RegionName := "output"
  let BLOCK_SIZE : Nat := blockSize
  with_triton_call callSignature call_args [n_elements := nElements] in
  triton {
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)
  }

end VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels
