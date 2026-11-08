import bench.examples.OnlineSoftmax.Proofs.RealSource
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core
import VeriTile.Triton.DSL
import VeriTile.Triton.Float
import VeriTile.Triton.KernelLemmas
import VeriTile.Triton.Math.Softmax
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Semantics
import bench.examples.OnlineSoftmax.Kernels
import VeriTile.Triton.Float.Kernel

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.OnlineSoftmax

open VeriTile.Bench.Examples.OnlineSoftmax.Kernels

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

open VeriTile.Examples

section OnlineSoftmax.kernelIO

open VeriTile.Triton.KernelIO₁ (Implements)

open scoped VeriTile.Triton.KernelIO₁

/-- Read and write the row at pid * B; input and output windows both have B elements. -/
def batchSoftmaxIO (B : Nat) : KernelIO₁ where
  kernel := (batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B).eraseDType
  projection := by
    simp only [batchSoftmaxKernel, RealSource.stable_projection]
    exact RealSource.stable_projectable "x" "y" B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The complete two-pass source has the same bounded row windows as batch softmax. -/
def onlineSoftmaxIO (N : Nat) : KernelIO₁ where
  kernel := (onlineSoftmaxKernel "x" "y" N).eraseDType
  projection := by
    rw [RealSource.online_projection]
    exact RealSource.online_projectable "x" "y" N
  inp := "x"
  out := "y"
  Bin := N
  Bout := N
  read := fun pid => pid * N
  write := fun pid => pid * N

end OnlineSoftmax.kernelIO

end VeriTile.Bench.Examples.OnlineSoftmax

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv

open VeriTile.Bench.Examples.OnlineSoftmax.Kernels

open VeriTile Triton

def batchIO (x y : RegionName) (N : Nat) : KernelIO₁ where
  kernel := stableSoftmaxKernel x y N
  inp := x
  out := y
  Bin := N
  Bout := N
  read := fun pid => pid * N
  write := fun pid => pid * N

end VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
