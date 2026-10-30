import bench.examples.FlatVectorAdd.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only snapshots of the previous complete programs. The current
Kernels.lean remains authoritative; these definitions only certify preservation. -/
namespace VeriTile.Bench.Examples.FlatVectorAdd.Kernels.SourcePreservation
open VeriTile Triton

def previous_addKernelMasked (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask    := offsets < $(nElements)
  x       := tl.load($(xReg) + offsets, mask=mask, dtype=tl.float32)
  y       := tl.load($(yReg) + offsets, mask=mask, dtype=tl.float32)
  output  := x + y
  tl.store($(outReg) + offsets, output, mask=mask)
}

/- The shared fp32 source with the example's input/output region names. -/
def previous_originalKernel (nElements blockSize : Nat) : ComputeKernel :=
  previous_addKernelMasked "x" "y" "out" blockSize nElements

/- The sole rewrite is output = y + x. Addresses/masks are unchanged. -/
def previous_optimizedKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offsets = pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = y + x
  tl.store(out_ptr + offsets, output, mask=mask)
}


theorem original (x y z : RegionName) (B n : Nat) : addKernelMasked x y z B n = previous_addKernelMasked x y z B n := rfl
theorem candidate (n B : Nat) : optimizedKernel n B = previous_optimizedKernel n B := rfl

#axiomsClean original
#axiomsClean candidate
end VeriTile.Bench.Examples.FlatVectorAdd.Kernels.SourcePreservation
