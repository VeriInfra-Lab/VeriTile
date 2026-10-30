import bench.examples.TritonBenchVectorAddition.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only snapshots of the previous complete programs. The current
Kernels.lean remains authoritative; these definitions only certify preservation. -/
namespace VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels.SourcePreservation
open VeriTile Triton

def previous_originalKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let output_ptr : Region .fp32 := ⟨"output"⟩
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(blockSize)
  offsets = block_start + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
}

/- The sole rewrite is output = y + x. Addresses/masks are unchanged. -/
def previous_optimizedKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let output_ptr : Region .fp32 := ⟨"output"⟩
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(blockSize)
  offsets = block_start + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = y + x
  tl.store(output_ptr + offsets, output, mask=mask)
}


theorem original (n B : Nat) : originalKernel n B = previous_originalKernel n B := rfl
theorem candidate (n B : Nat) : optimizedKernel n B = previous_optimizedKernel n B := rfl

#axiomsClean original
#axiomsClean candidate
end VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels.SourcePreservation
