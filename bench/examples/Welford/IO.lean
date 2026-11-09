import VeriTile.Examples.ErasedProjection
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit
import VeriTile.Triton
import bench.examples.Welford.Kernels
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.WelfordCorrect

open VeriTile.Bench.Examples.Welford.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.KernelIO₁ₓ₂

def twopassIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (twopassWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

def onlineIO (N stride : Nat) : KernelIO₁ₓ₂ :=
  (twopassIO N stride).withKernel (onlineWelfordKernel "x" "mean" "var" N stride).eraseDType
    (ComputeKernel.eraseDType_projectable _ (by rfl))

end VeriTile.Bench.Examples.WelfordCorrect

namespace VeriTile.Bench.Examples.WelfordFPExecution

open VeriTile.Bench.Examples.Welford.Kernels

open VeriTile Triton

/-- The original implementations expose both statistics, with symbolic row
length and stride. These contracts add no numerical relation between them. -/
def onlineIO (x mean variance : RegionName) (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := onlineWelfordKernel x mean variance N stride
  inp := x
  out1 := mean
  out2 := variance
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

def twopassIO (x mean variance : RegionName) (N stride : Nat) : KernelIO₁ₓ₂ :=
  (onlineIO x mean variance N stride).withKernel (twopassWelfordKernel x mean variance N stride)

end VeriTile.Bench.Examples.WelfordFPExecution

namespace VeriTile.Bench.Examples.WelfordFPEquiv

abbrev onlineIO := WelfordFPExecution.onlineIO
abbrev twopassIO := WelfordFPExecution.twopassIO

end VeriTile.Bench.Examples.WelfordFPEquiv
