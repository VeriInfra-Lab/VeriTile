import bench.examples.VectorAdd.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only snapshots of the previous complete programs. The current
Kernels.lean remains authoritative; these definitions only certify preservation. -/
namespace VeriTile.Bench.Examples.VectorAdd.Kernels.SourcePreservation
open VeriTile Triton

def previous_addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := tl.load($(yReg) + offs, dtype=tl.float32)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/- The shared fp32 source with the example's input/output region names. -/
def previous_originalKernel (blockSize : Nat) : ComputeKernel :=
  previous_addKernel "x" "y" "out" blockSize

/- The sole rewrite is out = y + x. Addresses/masks are unchanged. -/
def previous_optimizedKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offs = pid * $(blockSize) + tl.arange(0, $(blockSize))
  x = tl.load(x_ptr + offs)
  y = tl.load(y_ptr + offs)
  out = y + x
  tl.store(out_ptr + offs, out)
}


theorem original (x y z : RegionName) (B : Nat) : addKernel x y z B = previous_addKernel x y z B := rfl
theorem candidate (B : Nat) : optimizedKernel B = previous_optimizedKernel B := rfl

#axiomsClean original
#axiomsClean candidate
end VeriTile.Bench.Examples.VectorAdd.Kernels.SourcePreservation
