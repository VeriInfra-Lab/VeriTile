import VeriTile.Examples.ErasedProjection
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit
import VeriTile.Triton
import VeriTile.Triton.Math.Softmax
import bench.examples.SoftmaxReciprocal.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.Reciprocal

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.SoftmaxReciprocalCorrect

open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels

open VeriTile Triton Triton.TiledSoftmax

open scoped VeriTile.Triton.KernelIO₁

def divIO (B : Nat) : KernelIO₁ where
  kernel := (stableSoftmaxKernel "x" "y" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def recipIO (B : Nat) : KernelIO₁ where
  kernel := (softmaxRecipKernel "x" "y" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

end VeriTile.Bench.Examples.SoftmaxReciprocalCorrect

namespace VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv

open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels

open VeriTile Triton

open FP.Structural FP.Guarded

set_option maxHeartbeats 1600000

def typedIO (B : Nat) : KernelIO₁ where
  kernel := originalKernel "x" "y" B
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
