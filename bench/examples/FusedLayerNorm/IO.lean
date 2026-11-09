import VeriTile.Examples.ErasedProjection
import VeriTile.Meta.StatementAudit
import bench.examples.FusedLayerNorm.Kernels
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.FusedLayerNormCorrect

open VeriTile.Bench.Examples.FusedLayerNorm.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.KernelIO₃

/-- The real interpretation retains the original row and feature-array layout. -/
def twoPassIO (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := (twoPassLayerNormKernel "x" "gamma" "beta" "y" N stride ε).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "gamma"
  in3 := "beta"
  out := "y"
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

def fusedIO (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := (fusedLayerNormKernel "x" "gamma" "beta" "y" N stride ε).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "gamma"
  in3 := "beta"
  out := "y"
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

end VeriTile.Bench.Examples.FusedLayerNormCorrect

namespace VeriTile.Bench.Examples.LayerNormFPExecution

open VeriTile.Bench.Examples.FusedLayerNorm.Kernels

open VeriTile Triton

/-- Bind each input port once. The fused source reads `x` again for the affine
output and uses the same port and layout for both reads. -/
def twopassIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ where
  kernel := .mk [x, g, b] [y] (twoPassLayerNormKernel x g b y N stride ε).surfaceBody
  in1 := x
  in2 := g
  in3 := b
  out := y
  B1 := N
  B2 := N
  B3 := N
  Bout := N
  read1 := fun pid => pid * stride
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun pid => pid * stride

def fusedIO (x g b y : RegionName) (N stride : Nat) (ε : ℝ) : KernelIO₃ :=
  (twopassIO x g b y N stride ε).withKernel
    (.mk [x, g, b] [y] (fusedLayerNormKernel x g b y N stride ε).surfaceBody)

end VeriTile.Bench.Examples.LayerNormFPExecution

namespace VeriTile.Bench.Examples.FusedLayerNormFPEquiv

abbrev twopassIO := LayerNormFPExecution.twopassIO
abbrev fusedIO := LayerNormFPExecution.fusedIO

end VeriTile.Bench.Examples.FusedLayerNormFPEquiv
