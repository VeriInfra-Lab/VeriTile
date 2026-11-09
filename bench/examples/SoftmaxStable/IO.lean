import VeriTile.Examples.ErasedProjection
import VeriTile.Meta.StatementAudit
import bench.examples.SoftmaxStable.Kernels
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.Kernel

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.SoftmaxStableCorrect

open VeriTile.Bench.Examples.SoftmaxStable.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.KernelIO₁

def naiveIO (B : Nat) : KernelIO₁ where
  kernel := (naiveSoftmaxKernel "x" "y" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def stableIO (B : Nat) : KernelIO₁ where
  kernel := (stableSoftmaxKernel "x" "y" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

end VeriTile.Bench.Examples.SoftmaxStableCorrect

namespace VeriTile.Bench.Examples.SoftmaxStableFPEquiv

open VeriTile.Bench.Examples.SoftmaxStable.Kernels

open VeriTile Triton FP.Structural FP.Exponential

open scoped VeriTile.Spec

open _root_.VeriTile.Triton.FP.Equational (Schedules)

open FP.GuardExpression

def naiveIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := naiveSoftmaxKernel x y B
  inp := x
  out := y
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def stableIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  (naiveIO x y B).withKernel (stableSoftmaxKernel x y B)

end VeriTile.Bench.Examples.SoftmaxStableFPEquiv
