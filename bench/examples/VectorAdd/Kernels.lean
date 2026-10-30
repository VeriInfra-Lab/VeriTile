import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-!
VectorAdd: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.VectorAdd.Kernels
open VeriTile Triton

def originalSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32),("outReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def optimizedSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

/-- Elementwise add of two `blockSize`-element tiles, single-block.

Reads from `xReg` and `yReg`, writes to `outReg`. No aliasing assumption
required: even if `xReg = outReg` (read-modify-write of the same buffer)
the kernel reads first into local registers `x` / `y` before the final
scatter to `outReg`, so the result is still `xs + ys`. -/
def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature originalSignature in
  triton {
  pid  = tl.program_id(0)
  offs = pid * blockSize + tl.arange(0, blockSize)
  x    = tl.load(xReg + offs)
  y    = tl.load(yReg + offs)
  out  = x + y
  tl.store(outReg + offs, out)
}

/-- The shared fp32 source with the example's input/output region names. -/
def originalKernel (blockSize : Nat) : ComputeKernel :=
  addKernel "x" "y" "out" blockSize

/-- The sole rewrite is out = y + x. Addresses/masks are unchanged. -/
def optimizedKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let y_ptr : RegionName := "y"
  let out_ptr : RegionName := "out"
  with_triton_signature optimizedSignature in
  triton {
  pid = tl.program_id(axis=0)
  offs = pid * blockSize + tl.arange(0, blockSize)
  x = tl.load(x_ptr + offs)
  y = tl.load(y_ptr + offs)
  out = y + x
  tl.store(out_ptr + offs, out)
}

end VeriTile.Bench.Examples.VectorAdd.Kernels
