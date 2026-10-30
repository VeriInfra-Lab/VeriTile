import bench.examples.FusedSiLU.Kernels
import VeriTile.Triton.Float.Kernel

/-! Shared full contiguous per-program IO. The staged program writes two
private FP32 scratch tiles; the fused program writes only its BF16 output. -/
namespace VeriTile.Bench.Examples.FusedSiLU.IO
open VeriTile Triton
open VeriTile.Bench.Examples.FusedSiLU.Kernels

def unfusedIO (B : Nat) : KernelIO₃ where
  kernel := unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [{ buf := "z", win := fun pid => pid * B, len := B },
    { buf := "silu", win := fun pid => pid * B, len := B }]

def fusedIO (B : Nat) : KernelIO₃ :=
  { (unfusedIO B).withKernel (fusedSiLUKernel "x" "gate" "residual" "out" B) with
    scratch := [] }

end VeriTile.Bench.Examples.FusedSiLU.IO
