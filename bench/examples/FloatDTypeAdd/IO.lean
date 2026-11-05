import VeriTile.Examples.ErasedProjection
import VeriTile.Meta.StatementAudit
import bench.examples.FloatDTypeAdd.Kernels
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.FloatDTypeAddCorrect

open VeriTile.Bench.Examples.FloatDTypeAdd.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.KernelIO₂

/-- Correctness uses the mathematical projection of the original kernel. -/
def floatAddIO (B : Nat) : KernelIO₂ where
  kernel := (floatAddKernel "x" "y" "out" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B

/-- Keep the optimized fp32 source; erase quantization only for this real specification. -/
def optimizedIO (B : Nat) : KernelIO₂ :=
  { floatAddIO B with
    kernel := (optimizedKernel B).eraseDType
    projection := ComputeKernel.eraseDType_projectable _ (by rfl) }

end VeriTile.Bench.Examples.FloatDTypeAddCorrect
