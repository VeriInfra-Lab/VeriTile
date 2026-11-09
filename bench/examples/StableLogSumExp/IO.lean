import VeriTile.Examples.ErasedProjection
import VeriTile.Meta.StatementAudit
import bench.examples.StableLogSumExp.Kernels
import VeriTile.Triton.Float.Kernel

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.StableLogSumExpCorrect

open VeriTile.Bench.Examples.StableLogSumExp.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.KernelIO₁

def directIO (B : Nat) : KernelIO₁ where
  kernel := (directLSEKernel "x" "y" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

/-- Mathematical interpretation of the exact candidate source. -/
def optimizedIO (B : Nat) : KernelIO₁ where
  kernel := (optimizedLSEKernel "x" "y" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

end VeriTile.Bench.Examples.StableLogSumExpCorrect

noncomputable section

namespace VeriTile.Bench.Examples.StableLogSumExpFPEquiv

open VeriTile.Bench.Examples.StableLogSumExp.Kernels

open VeriTile Triton

open scoped VeriTile.Spec

open _root_.VeriTile.Triton.FP.Equational (Schedules)

open FP.GuardExpression

def directIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := directLSEKernel x y B
  inp := x
  out := y
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

def optimizedIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  (directIO x y B).withKernel (optimizedLSEKernel x y B)

end VeriTile.Bench.Examples.StableLogSumExpFPEquiv
